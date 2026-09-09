# # src/labor_rag/ingestion/loaders/web_loader.py

# import hashlib

# from selenium import webdriver
# from selenium.common.exceptions import TimeoutException, WebDriverException
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import WebDriverWait

# from labor_rag.models.document import RawDocument


# class SeleniumWebLoader:
#     """Mở URL bằng Chrome và lấy nội dung văn bản đang hiển thị."""

#     def __init__(
#         self,
#         headless: bool = True,
#         timeout: int = 20,
#     ) -> None:
#         self.headless = headless
#         self.timeout = timeout

#     def load(self, url: str) -> RawDocument:
#         """Nhận URL và trả về nội dung văn bản của trang."""

#         options = self._create_options()

#         try:
#             with webdriver.Chrome(options=options) as driver:
#                 driver.get(url)

#                 wait = WebDriverWait(driver, self.timeout)

#                 # Chờ tài liệu HTML tải xong.
#                 wait.until(
#                     lambda browser: browser.execute_script(
#                         "return document.readyState"
#                     )
#                     == "complete"
#                 )

#                 # Chờ phần body xuất hiện.
#                 body = wait.until(
#                     EC.visibility_of_element_located(
#                         (By.TAG_NAME, "body")
#                     )
#                 )

#                 title = driver.title.strip() or None

#                 # body.text chỉ lấy phần chữ đang hiển thị trên trang.
#                 content = body.text

#                 content = self._normalize_text(content)

#                 if not content:
#                     raise ValueError(
#                         f"Không lấy được nội dung từ URL: {url}"
#                     )

#                 return RawDocument(
#                     document_id=self._create_document_id(url),
#                     content=content,
#                     source_url=driver.current_url,
#                     title=title,
#                     metadata={
#                         "requested_url": url,
#                         "final_url": driver.current_url,
#                         "loader": "selenium",
#                     },
#                 )

#         except TimeoutException as exc:
#             raise RuntimeError(
#                 f"Trang tải quá thời gian cho phép: {url}"
#             ) from exc

#         except WebDriverException as exc:
#             raise RuntimeError(
#                 f"Không thể tải trang web: {url}"
#             ) from exc

#     def _create_options(self) -> Options:
#         options = Options()

#         if self.headless:
#             options.add_argument("--headless=new")
#         else:
#             options.add_argument("--start-maximized")

#         options.add_argument("--disable-gpu")
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")
#         options.add_argument("--window-size=1920,1080")

#         options.add_argument(
#             "--user-agent="
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/125.0.0.0 Safari/537.36"
#         )

#         return options

#     @staticmethod
#     def _normalize_text(text: str) -> str:
#         """Loại dòng trống và khoảng trắng thừa."""

#         lines = [
#             " ".join(line.split())
#             for line in text.splitlines()
#             if line.strip()
#         ]

#         return "\n".join(lines)

#     @staticmethod
#     def _create_document_id(url: str) -> str:
#         return hashlib.sha256(
#             url.encode("utf-8")
#         ).hexdigest()[:16]