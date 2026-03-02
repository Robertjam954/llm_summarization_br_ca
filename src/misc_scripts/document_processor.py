from __future__ import annotations

from typing import Any


class LuciDocumentProcessor:
    def __init__(self, file_obj: Any):
        self.file_obj = file_obj

    def get_text(self) -> str:
        if self.file_obj is None:
            return ""

        # Streamlit uploads, file paths, raw bytes, or file-like objects.
        if isinstance(self.file_obj, str):
            return self._read_path(self.file_obj)

        if isinstance(self.file_obj, (bytes, bytearray)):
            return self._extract_text_from_bytes(bytes(self.file_obj), filename=None)

        # file-like
        name = getattr(self.file_obj, "name", None)
        read = getattr(self.file_obj, "read", None)
        if callable(read):
            data = read()
            if isinstance(data, str):
                return data
            if isinstance(data, (bytes, bytearray)):
                return self._extract_text_from_bytes(bytes(data), filename=name)

        return str(self.file_obj)

    def _read_path(self, path: str) -> str:
        with open(path, "rb") as f:
            data = f.read()
        return self._extract_text_from_bytes(data, filename=path)

    def _extract_text_from_bytes(self, data: bytes, filename: str | None) -> str:
        ext = (filename or "").lower()

        # Try PDF
        if ext.endswith(".pdf"):
            try:
                from pypdf import PdfReader

                from io import BytesIO

                reader = PdfReader(BytesIO(data))
                pages_text = []
                for page in reader.pages:
                    pages_text.append(page.extract_text() or "")
                return "\n".join(pages_text).strip()
            except Exception:
                pass

        # Fallback: treat as UTF-8 text
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="ignore")
