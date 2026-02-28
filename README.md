# wikicom

GitHub Wiki를 자동으로 클론하고, 여러 .md 파일을 단일 Markdown 문서로 통합합니다.
내부 Wiki 링크를 문서 내부 앵커(`#anchor`)로 자동 변환합니다.
병합된 .md를 TOC 포함 HTML로 변환하고, PDF로 출력합니다.

```
$ wikicom clone https://github.com/user/repo.wiki.git
Cloning https://github.com/user/repo.wiki.git ...
Cloning into 'wikispace/repo.wiki'...

Cloned:    C:\project\wikispace\repo.wiki
Config:    C:\project\wikispace\repo.wiki\wikicom.yaml
Gitignore: C:\project\wikispace\repo.wiki\.gitignore  (wikicom.yaml 제외)

Next step:
  wikicom wikispace/repo.wiki

$ wikicom wikispace/repo.wiki

완료: 9개 파일 → C:\project\mergespace\repo.wiki.md
```

---

## Installation

### Option A — Python 스크립트로 직접 실행

```
pip install pyyaml markdown playwright
playwright install chromium
python wikicom.py clone https://github.com/user/repo.wiki.git
```

### Option B — 독립 실행 파일(.exe) 빌드

```
build.bat
```

빌드 후 설치 여부를 묻습니다. `y`를 입력하면 `%USERPROFILE%\tools\wikicom.exe`로 복사하고
시스템 PATH에 자동 등록합니다. 이후 어디서든 `wikicom` 명령으로 실행 가능합니다.

---

## Usage

```
wikicom clone <url> [--dir DIR]
wikicom pull <wiki-dir> [--force]
wikicom [merge] <wiki-dir> [options]
wikicom render [INPUT] [options]
wikicom pdf [INPUT] [options]
```

### clone 옵션

| Option | Description |
|--------|-------------|
| `URL` | GitHub wiki clone URL (예: `https://github.com/user/repo.wiki.git`) |
| `--dir DIR` | wikispace 부모 디렉토리 (기본: `./wikispace`) |

### pull 옵션

| Option | Description |
|--------|-------------|
| `WIKI_DIR` | 업데이트할 wiki 클론 디렉토리 |
| `--force` | 충돌 무시 후 원격 상태로 강제 동기화 (git fetch + reset --hard) |

### merge 옵션

| Option | Description |
|--------|-------------|
| `WIKI_DIR` | 클론된 wiki 디렉토리 (`merge` 키워드 생략 가능) |
| `--output FILE` | 출력 파일 경로 (기본: `mergespace/{wiki-folder-name}.md`) |
| `--config FILE` | `wikicom.yaml` 경로 명시 |
| `--base-url URL` | wiki 기준 URL (wikicom.yaml이 없을 때 필요) |
| `--page-break STR` | 파일 간 구분자 (기본: `---`, PDF용: `\newpage`) |
| `--bump-headings` | 헤딩 레벨 1 증가 (`#` → `##`) |
| `--home-file FILE` | 파일 순서 감지용 홈 파일 (기본: `Home.md`) |
| `--no-auto-order` | Home.md 기반 순서 감지 비활성화 (알파벳순) |
| `--verbose` | 링크 변환 내역 및 상세 출력 |
| `--dry-run` | 파일 쓰기 없이 stdout에 출력 |

### render 옵션

| Option | Description |
|--------|-------------|
| `INPUT` | 변환할 .md 파일 또는 디렉토리 (기본: `./mergespace/`) |
| `--output PATH` | 출력 경로 (기본: `./htmlspace/`) |
| `--title TITLE` | HTML `<title>` 태그 내용 (기본: 파일명) |
| `--css FILE` | 적용할 .css 파일 (기본: 내장 GitHub 스타일) |
| `--no-toc` | 목차(TOC) 생성 비활성화 |

### pdf 옵션

| Option | Description |
|--------|-------------|
| `INPUT` | 변환할 .html 파일 또는 디렉토리 (기본: `./htmlspace/`) |
| `--output PATH` | 출력 경로 (기본: `./pdfspace/`) |
| `--format FORMAT` | 페이지 크기 (기본: `A4`, 예: `Letter`, `A3`) |

## Examples

```bash
# GitHub wiki 클론 (wikispace/ 폴더 자동 생성)
wikicom clone https://github.com/user/repo.wiki.git

# 다른 위치에 클론
wikicom clone https://github.com/user/repo.wiki.git --dir ./docs

# wiki 최신 상태로 업데이트
wikicom pull wikispace/repo.wiki

# 병합 (merge 키워드 생략 가능) → mergespace/repo.wiki.md 생성
wikicom wikispace/repo.wiki

# 출력 경로 직접 지정
wikicom merge wikispace/repo.wiki --output ~/Desktop/wiki.md

# PDF용 페이지 구분자 + 헤딩 레벨 증가
wikicom merge wikispace/repo.wiki --page-break "\\newpage" --bump-headings

# 파일 쓰기 없이 결과 확인
wikicom merge wikispace/repo.wiki --dry-run

# 병합된 .md → HTML 변환
wikicom render

# HTML → PDF 변환
wikicom pdf

# 전체 워크플로우 (한 줄씩)
wikicom clone https://github.com/user/repo.wiki.git
wikicom merge wikispace/repo.wiki
wikicom render
wikicom pdf
```

---

## Workflow

1. **Clone** — wiki URL을 주면 자동으로 클론하고 `wikicom.yaml`을 생성합니다.
2. **Edit** (선택) — `wikispace/repo.wiki/wikicom.yaml`을 열어 필요한 옵션을 수정합니다.
3. **Merge** — 폴더 경로만 지정하면 yaml을 자동으로 찾아 병합을 실행합니다.
4. **Render** — 병합된 .md를 TOC 포함 HTML로 변환합니다.
5. **PDF** — HTML을 PDF로 변환합니다.

```
wikispace/
└── repo.wiki/           ← git clone 결과
    ├── wikicom.yaml     ← clone 시 자동 생성
    ├── Home.md
    └── ...

mergespace/
└── repo.wiki.md         ← merge 실행 후 생성

htmlspace/
└── repo.wiki.html       ← render 실행 후 생성

pdfspace/
└── repo.wiki.pdf        ← pdf 실행 후 생성
```

---

## wikicom.yaml 설정

```yaml
# 필수: GitHub Wiki 기준 URL (내부 링크 변환에 사용)
base_url: "https://github.com/user/repo/wiki"

# 출력 파일 경로 (기본: mergespace/{wiki-folder-name}.md)
# output: "mergespace/repo.wiki.md"

# 파일 간 구분자 (PDF용: "\\newpage")
page_break: "\n\n---\n\n"

# 파일 순서 자동 감지용 홈 파일
home_file: "Home.md"

# 헤딩 레벨 일괄 증가
bump_headings: false

# 병합에서 제외할 파일
exclude: []

# 파일별 세부 설정
# per_file:
#   Page-Name.md:
#     inject_heading: "# Page Title"   # 레벨-1 헤딩 없는 파일에 삽입
```

파일 순서는 `Home.md`의 내부 링크 순서를 자동으로 감지합니다.
`Home.md`에 없는 파일은 알파벳순으로 뒤에 추가됩니다.
명시적으로 순서를 지정하려면 `files:` 목록을 사용하세요.
