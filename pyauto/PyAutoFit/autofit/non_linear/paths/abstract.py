import atexit
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
import zipfile
import zlib
from abc import ABC, abstractmethod
from configparser import NoSectionError
from pathlib import Path
from typing import Optional

import numpy as np

from autonerves import conf
from autonerves.test_mode import is_test_mode
from autofit.mapper.identifier import Identifier, IdentifierField
from autofit.non_linear.samples.summary import SamplesSummary

from autofit.text import text_util
from autofit.tools.util import open_, zip_directory

logger = logging.getLogger(__name__)


_REPLAY_DIRS = set()


def _replay_dir_for(base_output_path) -> Path:
    """The folder this process replays the finished archive at ``base_output_path`` into."""
    return Path(f"{base_output_path}.replay-{socket.gethostname()}-{os.getpid()}")


def _cleanup_replay_dirs():
    """Drop every private replay folder this process extracted (registered with ``atexit``)."""
    while _REPLAY_DIRS:
        shutil.rmtree(_REPLAY_DIRS.pop(), ignore_errors=True)


atexit.register(_cleanup_replay_dirs)


def _remove_dead_replay_dirs(base_output_path):
    """
    Drop the private replay folders left beside ``base_output_path`` by processes of this
    host that no longer exist (a job killed before ``atexit`` could run).
    """
    host = socket.gethostname()
    for folder in Path(base_output_path).parent.glob(f"{Path(base_output_path).name}.replay-{host}-*"):
        try:
            pid = int(folder.name.rsplit("-", 1)[-1])
            os.kill(pid, 0)
        except (ValueError, IndexError):
            continue
        except ProcessLookupError:
            shutil.rmtree(folder, ignore_errors=True)
        except PermissionError:
            pass  # alive, another user's process


class SearchInProgressError(RuntimeError):
    """Another live process is fitting the same unfinished search (see ``_acquire_running_lock``)."""

pattern = re.compile(r"(?<!^)(?=[A-Z])")


def _test_mode_segment() -> Optional[str]:
    """
    Returns ``"test_mode"`` when ``PYAUTO_TEST_MODE`` has an active
    test-mode level, else ``None``.

    Inserted into the output-path composition so that smoke runs land
    under ``output/test_mode/...`` instead of sharing a directory with
    real runs. Without this, a cached test-mode result short-circuits
    a later real run at the same paths ("Fit Already Completed").
    """
    return "test_mode" if is_test_mode() else None


def _matches_archived(info: zipfile.ZipInfo, file_path) -> bool:
    """
    Returns whether the archived member described by ``info`` has the same
    content as the file at ``file_path``.

    Compared via the size and CRC already held in the zip's central directory,
    so the archived bytes are never decompressed and the file on disk is read
    once in chunks — a cache artifact can be a large ``.fits`` image.
    """
    if info.file_size != os.path.getsize(file_path):
        return False

    crc = 0
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)

    return crc == info.CRC


def _replace_zip_member(zip_path, arcname: str, file_path):
    """
    Replace the member ``arcname`` of the zip at ``zip_path`` with the file at
    ``file_path``.

    ``zipfile`` cannot overwrite a member in place, so every other member is
    streamed into a new archive alongside the replacement. The new archive is
    written next to the original (same directory, so the final ``os.replace``
    is atomic on one filesystem) and only swapped in once complete: an error
    part-way leaves the original archive untouched.

    Rebuilding the archive from the output directory instead (as the search
    itself does at completion) is not an option here, because with
    ``remove_files`` the directory holds only the file just written.
    """
    zip_path = Path(zip_path)
    temporary_path = zip_path.with_name(f"{zip_path.name}.replace.tmp")

    try:
        with zipfile.ZipFile(zip_path, "r") as source:
            replaced = source.getinfo(arcname)

            with zipfile.ZipFile(temporary_path, "w") as destination:
                for info in source.infolist():
                    if info.filename == arcname:
                        continue

                    with source.open(info) as member:
                        with destination.open(info, "w") as target:
                            shutil.copyfileobj(member, target)

                destination.write(
                    file_path,
                    arcname,
                    compress_type=replaced.compress_type,
                )

        os.replace(temporary_path, zip_path)
    except BaseException:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        raise


class AbstractPaths(ABC):
    stores_search_internal = True
    """
    Whether the search's internal state (``autofit.non_linear.checkpoint``) is
    stored on disk under ``search_internal/``. ``NullPaths`` (no output) and
    ``DatabasePaths`` (results live in the database) store none.
    """

    has_timer = True
    """
    Whether the search times itself into ``search_internal/`` (``Timer``).
    ``NullPaths`` and ``DatabasePaths`` do not.
    """

    supports_checkpointing = True
    """
    Whether a backend may write and resume from its own checkpoint file under these
    paths (Dynesty, Nautilus). ``NullPaths`` (no output) does not.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[os.PathLike] = None,
        is_identifier_in_paths=True,
        parent: Optional["AbstractPaths"] = None,
        unique_tag: Optional[str] = None,
        identifier: str = None,
        image_path_suffix: str = "",
    ):
        """
        Manages the path structure for `NonLinearSearch` output, for analyses both not using and using the search
        API. Use via non-linear searches requires manual input of paths, whereas the search API manages this using the
        search attributes.

        The output path within which the *Paths* objects path structure is contained is set via PyAutoNerves, using the
        command:

        from autonerves import conf
        conf.instance = conf.Config(output_path="path/to/output")

        If we assume all the input strings above are used with the following example names:

        name = "name"
        path_prefix = "folder_0/folder_1"

        The output path of the `NonLinearSearch` results will be:

        /path/to/output/folder_0/folder_1/name

        If ``PYAUTO_TEST_MODE`` has an active level greater than zero, a
        ``test_mode`` segment is inserted directly after the output root:

        /path/to/output/test_mode/folder_0/folder_1/name

        This keeps smoke-test artefacts in a sibling tree so they
        cannot be picked up by a later real run with the same paths.

        Parameters
        ----------
        name
            The name of the non-linear search, which is used as a folder name after the ``path_prefix``. For searchs
            this name is the ``name``.
        path_prefix
            A prefixed path that appears after the output_path but before the name variable.
        is_identifier_in_paths
            If True output path and symlink path terminate with an identifier generated from the
            search and model
        parent
            The parent paths object of this paths object.
        unique_tag
            A unique tag for the search, used to differentiate between searches with the same name.
        identifier
            A custom identifier for the search, if this is not None it will be used instead of the automatically
            generated identifier
        image_path_suffix
            A suffix which is appended to the image path. This is used to differentiate between different
            image outputs, for example the image of the starting point of an MLE.
        """

        self.name = name or ""
        self.path_prefix = path_prefix or ""

        self.unique_tag = unique_tag

        self._non_linear_name = None
        self.__custom_identifier = identifier
        self.__identifier = None

        self.is_identifier_in_paths = is_identifier_in_paths

        self._parent = None
        self.parent = parent

        try:
            self.remove_files = conf.instance["general"]["output"]["remove_files"]

            if conf.instance["general"]["hpc"]["hpc_mode"]:
                self.remove_files = True
        except NoSectionError as e:
            logger.exception(e)

        self.image_path_suffix = image_path_suffix

    @property
    @abstractmethod
    def samples(self):
        pass

    def save_parent_identifier(self):
        pass

    def save_unique_tag(self, is_grid_search=False):
        pass

    def __str__(self):
        return str(self.output_path)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    def parent(self) -> "AbstractPaths":
        """
        The search performed before this search. For example, a search
        that is then compared to searches during a grid search.
        """
        return self._parent

    @parent.setter
    @abstractmethod
    def parent(self, parent: "AbstractPaths"):
        pass

    @property
    @abstractmethod
    def is_grid_search(self) -> bool:
        pass

    def for_sub_analysis(self, analysis_name: str):
        return self.create_child(name=analysis_name)

    @abstractmethod
    def create_child(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        is_identifier_in_paths: Optional[bool] = None,
        identifier: Optional[str] = None,
    ) -> "AbstractPaths":
        """
        Create a paths object which is the child of some parent
        paths object. This is done during a GridSearch so that
        results can be stored in the correct directory.

        Parameters
        ----------
        name
        path_prefix
        is_identifier_in_paths
            If False then this path's identifier will not be
            added to its output path.
        identifier

        Returns
        -------
        A new paths object
        """

    search = IdentifierField()
    model = IdentifierField()
    unique_tag = IdentifierField()

    @property
    def non_linear_name(self):
        if self._non_linear_name is None:
            if self.search is not None:
                self._non_linear_name = pattern.sub(
                    "_", type(self.search).__name__
                ).lower()
        return self._non_linear_name

    @property
    def _identifier(self):
        if self.__custom_identifier is not None:
            return self.__custom_identifier
        if self.__identifier is None:
            if None in (self.model, self.search):
                logger.debug(
                    "Generating identifier without both model and search having been set."
                )

            identifier_list = [self.search, self.model]

            if self.unique_tag is not None:
                identifier_list.append(self.unique_tag)
            self.__identifier = Identifier(identifier_list)

        return self.__identifier

    @_identifier.setter
    def _identifier(self, identifier):
        self.__identifier = identifier

    @property
    def identifier(self):
        return str(self._identifier)

    def save_identifier(self):
        with open_(self.output_path / ".identifier", "w+") as f:
            f.write(self._identifier.description)

    @property
    def search_internal_path(self) -> Path:
        """
        The path to the samples folder.
        """

        os.makedirs(self._files_path / "search_internal", exist_ok=True)

        return self._files_path / "search_internal"

    @property
    def image_path(self) -> Path:
        """
        The path to the image folder.
        """

        if not (self.output_path / f"image{self.image_path_suffix}").exists():
            os.makedirs(self.output_path / f"image{self.image_path_suffix}")

        return self.output_path / f"image{self.image_path_suffix}"

    @property
    def profile_path(self) -> Path:
        """
        The path to the profile folder.
        """
        return self.output_path / "profile"

    @property
    def output_path(self) -> Path:
        """
        The path to the output information for a search.

        While a finished archive is being replayed (``restore`` found ``.completed`` in the
        zip and ``remove_files`` is set) this is the process-private folder the archive was
        extracted into, ``<base>.replay-<host>-<pid>``, for as long as that folder exists;
        every ``Paths`` of this process for the same search resolves to it, since callers
        (the sensitivity grid, expectation propagation) read a result through their own copy
        of the paths. The zip, the locks and the folder an unfinished fit writes stay at
        ``_base_output_path``.
        """
        base = self._base_output_path
        replay = _replay_dir_for(base)
        return replay if replay.exists() else base

    @property
    def _base_output_path(self) -> Path:
        strings = list(
            filter(
                None,
                [
                    str(conf.instance.output_path),
                    _test_mode_segment(),
                    str(self.path_prefix),
                    self.unique_tag,
                    str(self.name),
                ],
            )
        )

        if self.is_identifier_in_paths:
            strings.append(self.identifier)

        return Path(*strings) if strings else Path("")

    @property
    def _files_path(self) -> Path:
        """
        This is private for a reason, use the save_json etc. methods to save and load json
        """
        files_path = self.output_path / "files"
        try:
            os.makedirs(files_path, exist_ok=True)
        except FileExistsError:
            pass
        return files_path

    # Set by ``restore`` when the archive it extracted holds ``.completed``: the replay of a
    # finished fit is read-only (see ``restore``) and goes into a process-private folder
    # (``_replay_dir_for``), which ``output_path`` then points at until the process
    # exits (``_cleanup_replay_dirs``), so a second process replaying the same archive can
    # never delete a folder this one is still reading or writing its caches into.
    _archive_is_complete = False

    # Held from ``restore`` to ``_zip`` by the process fitting (starting or resuming) an
    # unfinished search; a second process finding it refuses to start (see
    # ``_acquire_running_lock``). A holder whose liveness cannot be checked is presumed dead
    # once its heartbeat (touched by ``touch_running_lock``) is older than this.
    _running_lock_held = False
    running_lock_stale_seconds = 6 * 3600

    # Whether this Paths owns an output folder on disk that the two locks above guard;
    # ``NullPaths`` and ``DatabasePaths`` do not.
    locks_output_folder = True

    def zip_remove(self):
        """
        Copy files from the sym linked search folder then remove the sym linked folder.
        """

        self._zip()

    def _zip(self):
        if self._archive_is_complete:
            # Read-only replay of a finished fit: the zip was never removed and nothing in
            # the extracted folder has changed (a cache written after completion goes into
            # the zip through ``preserve_in_zip``). The private extracted copy is kept until
            # the process exits: later stages still read from, and write caches into, this
            # result's folder, and the folder is this process's alone.
            self._archive_is_complete = False
            return

        try:
            zip_directory(self.output_path, self._zip_path)

            if self.remove_files:
                shutil.rmtree(
                    self.output_path,
                    ignore_errors=True,
                )

        except FileNotFoundError:
            pass
        finally:
            self._release_running_lock()

    @property
    def _running_lock_path(self) -> Path:
        return Path(f"{self._base_output_path}.running")

    def _acquire_running_lock(self):
        """
        Claim an unfinished search for this process, or refuse to touch it.

        A search that is starting or resuming writes its checkpoint and archive in place, so
        two processes on the same output folder corrupt each other (a job resubmitted while
        its predecessor still runs, or two benchmarks of one lens writing into one prefix).
        The lock is a directory beside the output folder holding the claimant's host, pid
        and SLURM job id. A second process finding one checks whether the holder is alive --
        its SLURM job through ``squeue`` when that exists, else its pid when on the same host,
        else its heartbeat against ``running_lock_stale_seconds`` -- and raises
        ``SearchInProgressError`` if so. A dead holder's lock is taken over with a warning,
        as is a lock left by the same SLURM job (a runner that re-executed itself after an
        out-of-memory retry).
        """
        if not self.locks_output_folder:
            return
        lock = self._running_lock_path
        lock.parent.mkdir(parents=True, exist_ok=True)
        holder = self._running_lock_holder()
        try:
            os.mkdir(lock)
        except FileExistsError:
            other = self._read_lock_holder(lock)
            if self._lock_holder_alive(other, holder):
                raise SearchInProgressError(
                    f"The search at {self.output_path} is being fitted by another process "
                    f"({other}); refusing to start on it. Give this run its own output "
                    f"prefix, or wait for that process to finish."
                )
            logger.warning(
                f"Taking over the running-search lock {lock} left by {other}, which is no "
                f"longer alive."
            )
        self._write_lock_holder(lock, holder)
        self._running_lock_held = True

    def _release_running_lock(self):
        if self._running_lock_held:
            shutil.rmtree(self._running_lock_path, ignore_errors=True)
            self._running_lock_held = False

    def touch_running_lock(self):
        """Refresh the heartbeat of the running-search lock this process holds (if any)."""
        if self._running_lock_held:
            try:
                os.utime(self._running_lock_path / "holder")
            except OSError:
                pass

    @staticmethod
    def _running_lock_holder() -> dict:
        return {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "time": time.ctime(),
        }

    @staticmethod
    def _write_lock_holder(lock: Path, holder: dict):
        try:
            (lock / "holder").write_text(json.dumps(holder))
        except OSError:
            pass

    @staticmethod
    def _read_lock_holder(lock: Path) -> dict:
        try:
            return json.loads((lock / "holder").read_text())
        except (OSError, ValueError):
            return {}

    def _lock_holder_alive(self, other: dict, me: dict) -> bool:
        """Whether the process described by ``other`` (a lock's holder file) still runs."""
        if not other:
            return self._lock_heartbeat_fresh()
        if other.get("host") == me.get("host") and other.get("pid") == me.get("pid"):
            return False  # this very process: an earlier fit here raised before releasing it
        job = other.get("slurm_job_id")
        if job and job == me.get("slurm_job_id"):
            return False  # the same SLURM job, re-executed: its own lock
        if job and shutil.which("squeue"):
            try:
                state = subprocess.run(
                    ["squeue", "-h", "-j", str(job), "-o", "%T"],
                    capture_output=True, text=True, timeout=30,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                state = None
            if state is not None:
                return state not in ("", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT",
                                     "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "COMPLETING")
        if other.get("host") == me.get("host") and other.get("pid"):
            try:
                os.kill(int(other["pid"]), 0)
                return True
            except (OSError, ValueError):
                return False
        return self._lock_heartbeat_fresh()

    def _lock_heartbeat_fresh(self) -> bool:
        try:
            age = time.time() - (self._running_lock_path / "holder").stat().st_mtime
        except OSError:
            try:
                age = time.time() - self._running_lock_path.stat().st_mtime
            except OSError:
                return False
        return age < self.running_lock_stale_seconds

    def preserve_in_zip(self, file_path):
        """
        Add a file (already inside this search's output directory) to the
        search's ``.zip`` archive so it survives the resume cycle.

        ``restore()`` deletes the output directory and re-extracts the zip, so
        any file written into e.g. ``files/`` *after* a search completed — such
        as a cache artifact derived from the finished result — would be
        destroyed by the next resume unless it is also a member of the zip.

        No-op when the zip does not exist (e.g. the search is still running,
        so the file will be zipped with everything else at completion) and
        when the archived member is already byte-identical to the file on
        disk. When the member exists but its content has changed — a cache
        that was invalidated and recomputed — it is replaced, otherwise the
        stale copy would come back at the next ``restore()``.

        Under ``remove_files`` the zip is the **only** store for a search:
        the output directory is deleted once the search is archived, so a
        loose copy left beside the zip is not a second home for the file but
        a directory that should not exist. It shadows the zip for
        ``Aggregator.from_directory``, which then sees a search output with
        no ``.completed`` file. The loose copy is therefore deleted once the
        member is in the archive, along with any parent directory that
        emptied as a result, and the cache it holds is recomputed the next
        time it is asked for. With ``remove_files`` unset the output
        directory is the primary store and the loose copy stays.

        Parameters
        ----------
        file_path
            Absolute path of the file to preserve; must live under
            ``output_path``, from which its archive name is derived.
        """
        if not Path(self._zip_path).exists():
            return

        file_path = Path(file_path)
        arcname = str(file_path.relative_to(self.output_path))

        replace = False

        with zipfile.ZipFile(self._zip_path, "a") as f:
            try:
                info = f.getinfo(arcname)
            except KeyError:
                f.write(file_path, arcname)
                info = None

            if info is not None and not _matches_archived(info, file_path):
                replace = True

        if replace:
            _replace_zip_member(
                zip_path=self._zip_path,
                arcname=arcname,
                file_path=file_path,
            )

        if self.remove_files:
            self._remove_preserved_copy(file_path)

    def _remove_preserved_copy(self, file_path: Path):
        """
        Delete a file that has just been written into the search's zip, and any
        directory between it and ``output_path`` that the deletion emptied.

        Called only under ``remove_files``, where the zip is the search's only
        store; see ``preserve_in_zip``.

        Parameters
        ----------
        file_path
            Absolute path of the loose file, which lives under ``output_path``.
        """
        output_path = Path(self.output_path)

        try:
            file_path.unlink()
        except FileNotFoundError:
            return
        except OSError as e:
            logger.debug(f"Could not remove the preserved copy at {file_path}: {e}")
            return

        directory = file_path.parent
        while True:
            try:
                directory.relative_to(output_path)
            except ValueError:
                # Walked above the search's own output directory.
                return
            try:
                directory.rmdir()
            except OSError:
                # Not empty, or gone already — nothing further to prune.
                return
            if directory == output_path:
                return
            directory = directory.parent

    def restore(self):
        """
        Copy files from the ``.zip`` file to the samples folder.

        An archive of a *finished* fit (one holding ``.completed``) is replayed read-only:
        the zip stays in place and is extracted into a folder private to this process,
        ``<output>.replay-<host>-<pid>``, which ``output_path`` points at from then on
        and which is removed when the process exits. Before 2026-10-24 every replay removed
        the zip and re-created it from the folder, so two processes reading one finished
        search at the same time could leave a corrupt archive behind; a lock directory then
        serialised replays of a shared folder, but a stage that reads a result's files
        (PyAutoGalaxy's adapt-image cache) after its replay still found the folder deleted
        by the next process's replay (2026-10-26, four SLaM chains sharing one SOURCE LP).
        An archive of an unfinished fit is restored as before, into the folder beside the
        zip: the zip is removed and rebuilt when the resumed fit completes.
        """

        if not Path(self._zip_path).exists():
            # A fresh start, or a folder left by a run that never archived: claim it.
            if not self._running_lock_held:
                self._acquire_running_lock()
            return

        try:
            with zipfile.ZipFile(self._zip_path, "r") as f:
                complete = ".completed" in f.namelist()
        except zipfile.BadZipFile as e:
            raise zipfile.BadZipFile(
                f"Unable to restore the zip file at the path {self._zip_path}"
            ) from e

        if complete and self.remove_files:
            # A private folder per process (and per Paths): two processes replaying one
            # finished archive at the same time never touch each other's copy.
            _remove_dead_replay_dirs(self._base_output_path)
            _REPLAY_DIRS.add(_replay_dir_for(self._base_output_path))
        elif not complete and not self._running_lock_held:
            self._acquire_running_lock()

        target = (
            _replay_dir_for(self._base_output_path)
            if complete and self.remove_files
            else self._base_output_path
        )
        shutil.rmtree(target, ignore_errors=True)

        try:
            try:
                with zipfile.ZipFile(self._zip_path, "r") as f:
                    f.extractall(target)
            except FileExistsError:
                pass
        except zipfile.BadZipFile as e:
            raise zipfile.BadZipFile(
                f"Unable to restore the zip file at the path {self._zip_path}"
            ) from e

        self._archive_is_complete = complete

        if not complete:
            try:
                os.remove(self._zip_path)
            except FileNotFoundError:
                pass

    def __eq__(self, other):
        return isinstance(other, AbstractPaths) and all(
            [
                self.path_prefix == other.path_prefix,
                self.name == other.name,
                self.non_linear_name == other.non_linear_name,
            ]
        )

    @property
    def _zip_path(self) -> str:
        return f"{self._base_output_path}.zip"

    @abstractmethod
    def save_json(self, name, object_dict: dict, prefix: str = ""):
        pass

    @abstractmethod
    def load_json(self, name, prefix: str = "") -> dict:
        pass

    @abstractmethod
    def save_array(self, name, array: np.ndarray):
        pass

    @abstractmethod
    def load_array(self, name) -> np.ndarray:
        pass

    @abstractmethod
    def save_fits(self, name: str, fits, prefix: str = ""):
        pass

    @abstractmethod
    def load_fits(self, name: str, prefix: str = ""):
        pass

    @abstractmethod
    def save_object(self, name: str, obj: object, prefix: str = ""):
        pass

    @abstractmethod
    def load_object(self, name: str, prefix: str = ""):
        pass

    @abstractmethod
    def remove_object(self, name: str):
        pass

    @abstractmethod
    def is_object(self, name: str) -> bool:
        pass

    def save_search_internal(self, obj):
        raise NotImplementedError

    def load_search_internal(self):
        raise NotImplementedError

    def remove_search_internal(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @abstractmethod
    def completed(self):
        pass

    @abstractmethod
    def save_all(self, search_config_dict=None, info=None):
        pass

    @abstractmethod
    def load_samples(self):
        """
        Load samples from the database
        """

    @abstractmethod
    def save_samples(self, samples):
        """
        Save samples to the database
        """

    def save_samples_summary(
        self,
        samples_summary: SamplesSummary,
        name: str = "",
    ):
        """
        Save samples summary to the database.
        """

    def load_samples_summary(self) -> SamplesSummary:
        """
        Load samples summary from the database.
        """

    @abstractmethod
    def save_latent_samples(self, latent_samples):
        """
        Save latent variables. These are values computed from an instance and output
        during analysis.
        """

    @abstractmethod
    def load_samples_info(self):
        pass

    def save_summary(
        self,
        samples,
        latent_samples,
        log_likelihood_function_time,
        visualization_time = None,
    ):
        result_info = text_util.result_info_from(
            samples=samples,
        )
        self.output_model_results(result_info=result_info)

        if latent_samples:
            result_info = text_util.result_info_from(
                samples=latent_samples,
            )
            filename = self.output_path / "latent.results"
            with open_(filename, "w") as f:
                f.write(result_info)

        search = getattr(self, "search", None)
        search_capabilities = None
        if search is not None and getattr(type(search), "objective_target", None):
            from autofit.non_linear.search.capabilities import (
                capability_summary_from,
                run_summary_from,
            )

            search_capabilities = capability_summary_from(search) + run_summary_from(
                search
            )

        text_util.search_summary_to_file(
            samples=samples,
            log_likelihood_function_time=log_likelihood_function_time,
            visualization_time=visualization_time,
            filename=self.output_path / "search.summary",
            search_capabilities=search_capabilities,
        )

    @property
    def _samples_file(self) -> Path:
        return self._files_path / "samples.csv"

    @property
    def _latent_variables_file(self) -> Path:
        return self._files_path / "latent.csv"

    @property
    def _covariance_file(self) -> Path:
        return self._files_path / "covariance.csv"

    @property
    def _info_file(self) -> Path:
        return self._files_path / "samples_info.json"

    def output_model_results(self, result_info):

        filename = self.output_path / "model.results"

        with open_(filename, "w") as f:
            f.write(result_info)
