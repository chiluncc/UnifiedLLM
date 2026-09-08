import base64
import filetype
from pathlib import Path
from typing import Literal, override
from urllib.parse import urlsplit

from .base import ContentException, ContentHumanBase


class ContentHumanText(ContentHumanBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentHumanText":
        return ContentHumanText(self._text)


class ContentHumanImage(ContentHumanBase):
    def __init__(
        self,
        path: str | Path,
        *,
        detail: Literal["auto", "low", "high", "original"] = "auto",
    ) -> None:
        super().__init__()
        self._source: str
        self._mode: Literal["local", "remote"]
        self._path: Path | None
        self._type: str | None
        self._base64: str | None
        self._detail: Literal["auto", "low", "high", "original"] = detail

        if isinstance(path, Path):
            source = str(path)
            is_remote = False
        else:
            source = path
            is_remote = urlsplit(source).scheme in ("http", "https")

        if is_remote:
            self._mode = "remote"
            self._source = source
            self._path = None
            self._type = None
            self._base64 = None
            return

        file_path = Path(source)
        if not file_path.is_file():
            raise ContentException(
                f"ContentHumanImage can't init: file not found: {file_path}"
            )
        image_data = file_path.read_bytes()
        detected = filetype.guess(image_data)
        if detected is None or not detected.mime.startswith("image/"):
            raise ContentException(
                f"ContentHumanImage can't init: unsupported image file: {file_path}"
            )

        self._mode = "local"
        self._source = source
        self._path = file_path
        self._type = detected.mime
        self._base64 = base64.b64encode(image_data).decode("ascii")

    @classmethod
    def _from_loaded(
        cls,
        *,
        mode: Literal["local", "remote"],
        source: str,
        path: Path | None,
        img_type: str | None,
        img_base64: str | None,
        detail: Literal["auto", "low", "high", "original"],
    ) -> "ContentHumanImage":
        instance = cls.__new__(cls)
        instance._mode = mode
        instance._source = source
        instance._path = path
        instance._type = img_type
        instance._base64 = img_base64
        instance._detail = detail
        return instance
    
    @property
    def mode(self) -> Literal["local", "remote"]:
        return self._mode

    @property
    def img_path(self) -> Path | None:
        return self._path

    @property
    def img_type(self) -> str | None:
        return self._type

    @property
    def img_base64(self) -> str | None:
        return self._base64

    @property
    def img_url(self) -> str:
        if self._mode == "remote":
            return self._source
        return f"data:{self._type};base64,{self._base64}"

    @property
    def detail(self) -> Literal["auto", "low", "high", "original"]:
        return self._detail

    @override
    def copy(self) -> "ContentHumanImage":
        return ContentHumanImage._from_loaded(
            mode=self._mode,
            source=self._source,
            path=self._path,
            img_type=self._type,
            img_base64=self._base64,
            detail=self._detail,
        )