import base64
import filetype
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, override
from urllib.parse import urlsplit


class ContentException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class ContentBase(ABC):
    @abstractmethod
    def copy(self) -> "ContentBase": ...


class ContentSystemBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentSystemBase": ...


class ContentHumanBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentHumanBase": ...


class ContentAIBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentAIBase": ...


class ContentToolBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentToolBase": ...


################################


class ContentSystemText(ContentSystemBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentSystemText":
        return ContentSystemText(self._text)


################################


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


################################


class ContentAIReasoningBase(ContentAIBase, ABC):
    @override
    def copy(self) -> "ContentAIReasoningBase": ...


class ContentAIText(ContentAIBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentAIText":
        return ContentAIText(self._text)


class ContentAIToolCall(ContentAIBase):
    def __init__(self, tool_name: str, tool_args: str, tool_id: str):
        super().__init__()
        self._tool_name: str = tool_name
        self._tool_args: str = tool_args
        self._tool_id: str = tool_id

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def tool_args(self) -> str:
        return self._tool_args

    @property
    def tool_id(self) -> str:
        return self._tool_id

    @override
    def copy(self) -> "ContentAIToolCall":
        return ContentAIToolCall(
            tool_name=self._tool_name,
            tool_args=self._tool_args,
            tool_id=self._tool_id,
        )


################################


class ContentAIReasoningText(ContentAIReasoningBase):
    def __init__(self, *, reasoning_content: str | None = None, reasoning_summary: str | None = None):
        super().__init__()
        self._reasoning_content: str | None = reasoning_content
        self._reasoning_summary: str | None = reasoning_summary
        if self._reasoning_content is None and self._reasoning_summary is None:
            raise ContentException(
                "ContentReasoningText can't init: at least one of reasoning_content or reasoning_summary must be provided"
            )

    @property
    def reasoning_content(self) -> str | None:
        return self._reasoning_content

    @property
    def reasoning_summary(self) -> str | None:
        return self._reasoning_summary

    @override
    def copy(self) -> "ContentAIReasoningText":
        return ContentAIReasoningText(
            reasoning_content=self._reasoning_content,
            reasoning_summary=self._reasoning_summary,
        )


################################


class ContentToolText(ContentToolBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentToolText":
        return ContentToolText(self._text)


class ContentToolImage(ContentToolBase):
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
                f"ContentToolImage can't init: file not found: {file_path}"
            )
        image_data = file_path.read_bytes()
        detected = filetype.guess(image_data)
        if detected is None or not detected.mime.startswith("image/"):
            raise ContentException(
                f"ContentToolImage can't init: unsupported image file: {file_path}"
            )

        self._mode = "local"
        self._source = source
        self._path = file_path
        self._type = detected.mime
        self._base64 = base64.b64encode(image_data).decode("ascii")

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
    def copy(self) -> "ContentToolImage":
        return ContentToolImage._from_loaded(
            mode=self._mode,
            source=self._source,
            path=self._path,
            img_type=self._type,
            img_base64=self._base64,
            detail=self._detail,
        )

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
    ) -> "ContentToolImage":
        instance = cls.__new__(cls)
        instance._mode = mode
        instance._source = source
        instance._path = path
        instance._type = img_type
        instance._base64 = img_base64
        instance._detail = detail
        return instance
