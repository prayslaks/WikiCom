from playwright.sync_api import sync_playwright

def create_pdf(html_path, output_pdf):
    with sync_playwright() as p:
        # 브라우저 실행 (백그라운드)
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # 로컬 HTML 파일 또는 URL 열기
        page.goto(f"file://{html_path}", wait_until="networkidle")
        
        # PDF 저장 설정
        page.pdf(
            path=output_pdf,
            format="A4",
            print_background=True,  # 배경색/이미지 포함 여부
            prefer_css_page_size=True, # CSS에서 설정한 페이지 크기를 우선함
            margin={"top": "20px", "bottom": "20px", "left": "20px", "right": "20px"}
        )
        browser.close()

html_path = "path_to_target_html"
create_pdf(html_path, "result.pdf")