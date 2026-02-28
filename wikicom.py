#!/usr/bin/env python3
"""
wikicom — GitHub Wiki 클로너 & Markdown 병합기 & HTML 렌더러

GitHub Wiki를 자동으로 클론하고, 여러 .md 파일을 단일 .md 파일로 통합합니다.
내부 Wiki 링크를 문서 내부 앵커(#anchor)로 자동 변환합니다.
병합된 .md 파일을 TOC 포함 HTML로 변환합니다.

Usage:
    wikicom clone <url> [--dir DIR]
    wikicom pull <wiki-dir> [--force]
    wikicom [merge] <wiki-dir> [options]
    wikicom render [INPUT] [options]
    wikicom pdf [INPUT] [options]

Examples:
    wikicom clone https://github.com/user/repo.wiki.git
    wikicom clone https://github.com/user/repo.wiki.git --dir ./docs
    wikicom pull wikispace/repo.wiki
    wikicom pull wikispace/repo.wiki --force
    wikicom wikispace/repo.wiki
    wikicom merge wikispace/repo.wiki --output merged.md
    wikicom merge wikispace/repo.wiki --bump-headings --dry-run
    wikicom render
    wikicom render mergespace/repo.wiki.md
    wikicom render mergespace/ --output htmlspace/
    wikicom pdf
    wikicom pdf htmlspace/repo.wiki.html
    wikicom pdf htmlspace/ --output pdfspace/

Dependencies:
    pip install pyyaml markdown playwright
    playwright install chromium
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

VERSION = "1.0.0"

SUBCOMMANDS = frozenset({'clone', 'merge', 'pull', 'render', 'pdf'})

DEFAULT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
    line-height: 1.6;
    color: #24292e;
}
img {
    max-width: 100%;            /* 부모 너비를 절대 넘지 않음 */
    height: auto;               /* 가로 세로 비율 유지 */
    display: block;             /* 하단 여백 제거 및 레이아웃 안정화 */
    page-break-inside: avoid;   /* 페이지 중간에서 이미지가 잘리는 것 방지 */
}
h1, h2, h3, h4, h5, h6 { margin-top: 1.5em; margin-bottom: 0.5em; }
pre {
    white-space: pre-wrap;       /* 공백 유지하며 줄바꿈 */
    word-break: break-all;       /* 단어 단위와 상관없이 강제 줄바꿈 */
    overflow-wrap: break-word;   /* 긴 단어도 줄바꿈 허용 */
    background: #f6f8fa;
    padding: 1em;
    border-radius: 6px;
    overflow-x: auto;
}
code { 
    background: #f6f8fa; 
    padding: 0.2em 0.4em; 
    border-radius: 3px; 
    font-size: 80%;
    font-family: 'D2Coding', 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-variant-ligatures: contextual; /* 합자 기능 활성화 */ 
}
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #d1d5da; padding: 0.5em 1em; }
th { background: #f6f8fa; }
blockquote { margin: 0; padding: 0 1em; border-left: 4px solid #d1d5da; color: #6a737d; }
hr { border: none; border-top: 1px solid #e1e4e8; margin: 2em 0; }
a { color: #0366d6; }
.toc { background: #f6f8fa; padding: 1em 1.5em; border-radius: 6px; margin-bottom: 2em; }
.toc ul { margin: 0.3em 0; }
.page-break { page-break-before: always; break-before: page; }
.cover-page {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    page-break-after: always;
    break-after: page;
    padding: 4rem 2rem;
}
.cover-page h1 { font-size: 2.5em; margin: 0 0 0.5em; border-bottom: none; }
.cover-page p  { color: #6a737d; margin: 0.3em 0; }
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_warnings: List[str] = []


def _warn(message: str) -> None:
    """경고를 stderr에 출력하고 목록에 기록합니다."""
    full = f"WARNING: {message}"
    print(full, file=sys.stderr)
    _warnings.append(full)


def _error(message: str) -> None:
    """오류를 stderr에 출력하고 종료합니다."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class WikiPage:
    filename: str                 # "Authentication-Flow.md"
    slug: str                     # "Authentication-Flow"
    path: Path                    # 절대 경로
    raw_content: str              # 원본 파일 내용
    first_heading: Optional[str]  # "Authentication Flow" (마크다운 없이)
    anchor: str                   # "authentication-flow"


# ---------------------------------------------------------------------------
# Anchor Generation
# ---------------------------------------------------------------------------

def heading_to_anchor(text: str) -> str:
    """
    헤딩 텍스트로부터 pandoc 호환 앵커 ID를 생성합니다.

    예:
      "Getting Started — JWNetworkUtility" -> "getting-started-jwnetworkutility"
      "C++ Usage"                          -> "c-usage"
      "API Reference"                      -> "api-reference"
    """
    text = re.sub(r'[*_`~\[\]()]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', '-', text.strip())
    text = re.sub(r'[^\w-]', '', text)
    text = re.sub(r'-+', '-', text).strip('-')
    return text


def extract_heading_text(line: str) -> str:
    """'# My Heading' 또는 '  ## My Heading' 에서 순수 텍스트만 추출."""
    return re.sub(r'^\s{0,3}#+\s*', '', line).strip()


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'base_url': None,
    'output': None,           # None → wiki_dir/merged.md
    'home_file': 'Home.md',
    'files': None,            # None → Home.md에서 자동 감지
    'page_break': '\n\n---\n\n',
    'bump_headings': False,
    'exclude': [],
    'per_file': {},
    'cover': None,            # None → 표지 미생성
}


def load_config(wiki_dir: Path, cli_args) -> dict:
    """CLI 인자와 wikicom.yaml 설정 파일을 병합하여 최종 설정을 반환합니다."""
    cfg = dict(DEFAULT_CONFIG)
    cfg['output'] = str(Path.cwd() / 'mergespace' / (wiki_dir.name + '.md'))

    # 설정 파일 탐색: --config > WIKI_DIR/wikicom.yaml > CWD/wikicom.yaml
    config_path = None
    if getattr(cli_args, 'config', None):
        config_path = Path(cli_args.config)
        if not config_path.exists():
            _error(f"설정 파일을 찾을 수 없습니다: {config_path}")
    elif (wiki_dir / 'wikicom.yaml').exists():
        config_path = wiki_dir / 'wikicom.yaml'
    elif (Path.cwd() / 'wikicom.yaml').exists():
        config_path = Path.cwd() / 'wikicom.yaml'

    if config_path:
        try:
            import yaml
        except ImportError:
            _error("pyyaml이 설치되어 있지 않습니다. 'pip install pyyaml' 실행 후 재시도하세요.")

        with open(config_path, 'r', encoding='utf-8') as f:
            file_cfg = yaml.safe_load(f) or {}

        for key, val in file_cfg.items():
            if val is not None:
                cfg[key] = val

    # CLI 플래그로 덮어쓰기
    if getattr(cli_args, 'base_url', None):
        cfg['base_url'] = cli_args.base_url
    if getattr(cli_args, 'output', None):
        cfg['output'] = cli_args.output
    if getattr(cli_args, 'page_break', None):
        cfg['page_break'] = cli_args.page_break
    if getattr(cli_args, 'bump_headings', False):
        cfg['bump_headings'] = True
    if getattr(cli_args, 'home_file', None):
        cfg['home_file'] = cli_args.home_file
    if getattr(cli_args, 'no_auto_order', False):
        cfg['no_auto_order'] = True

    # page_break 이스케이프 처리
    pb = cfg['page_break']
    try:
        cfg['page_break'] = pb.encode('raw_unicode_escape').decode('unicode_escape')
    except (UnicodeDecodeError, ValueError):
        pass

    # base_url 후행 슬래시 제거
    if cfg['base_url']:
        cfg['base_url'] = cfg['base_url'].rstrip('/')

    # 유효성 검사
    if not cfg['base_url']:
        _error(
            "base_url이 설정되지 않았습니다.\n"
            "  wikicom.yaml에 base_url을 추가하거나,\n"
            "  --base-url 플래그로 지정하세요.\n"
            "  예: --base-url https://github.com/user/repo/wiki"
        )

    return cfg


# ---------------------------------------------------------------------------
# File Discovery & Scanning
# ---------------------------------------------------------------------------

def discover_files(wiki_dir: Path, config: dict) -> List[WikiPage]:
    """wiki_dir의 모든 .md 파일을 읽고 WikiPage 목록을 반환합니다."""
    exclude_set = set(config.get('exclude', []))
    per_file_cfg = config.get('per_file', {}) or {}

    pages = []
    for md_path in sorted(wiki_dir.glob('*.md')):
        filename = md_path.name
        if filename in exclude_set:
            continue

        slug = md_path.stem

        try:
            raw_content = md_path.read_text(encoding='utf-8')
        except OSError as e:
            _warn(f"{filename} — 파일 읽기 실패: {e}")
            continue

        file_cfg = per_file_cfg.get(filename, {}) or {}
        inject = file_cfg.get('inject_heading', None)

        first_heading = None
        for line in raw_content.splitlines():
            if re.match(r'^\s{0,3}#\s', line):
                first_heading = extract_heading_text(line)
                break

        anchor_source = first_heading
        if inject:
            injected_text = extract_heading_text(inject)
            anchor_source = injected_text
            if first_heading is None:
                first_heading = injected_text

        if anchor_source is None:
            _warn(
                f"{filename} — 레벨-1 헤딩(#)이 없습니다. "
                f"파일명에서 앵커를 생성합니다: '{slug}'\n"
                f"  해결: wikicom.yaml의 per_file.{filename}.inject_heading 설정"
            )
            anchor_source = slug.replace('-', ' ')

        anchor = heading_to_anchor(anchor_source)

        pages.append(WikiPage(
            filename=filename,
            slug=slug,
            path=md_path,
            raw_content=raw_content,
            first_heading=first_heading,
            anchor=anchor,
        ))

    return pages


def build_slug_anchor_map(pages: List[WikiPage]) -> Dict[str, str]:
    """
    {slug: anchor} 매핑을 생성합니다.
    예: {"Authentication-Flow": "authentication-flow"}
    """
    slug_map: Dict[str, str] = {}
    anchor_seen: Dict[str, str] = {}

    for page in pages:
        slug_map[page.slug] = page.anchor
        if page.anchor in anchor_seen:
            _warn(
                f"앵커 충돌: '{page.anchor}' 이 {anchor_seen[page.anchor]} 와 "
                f"{page.filename} 에서 동시에 생성됩니다."
            )
        else:
            anchor_seen[page.anchor] = page.filename

    return slug_map


# ---------------------------------------------------------------------------
# Order Resolution
# ---------------------------------------------------------------------------

def infer_order_from_home(
    pages: List[WikiPage],
    slug_anchor_map: Dict[str, str],
    config: dict,
    wiki_dir: Path,
    verbose: bool = False,
) -> List[WikiPage]:
    """
    Home.md의 내부 wiki 링크 순서를 기반으로 파일 순서를 결정합니다.
    Home.md에 없는 파일은 알파벳순으로 뒤에 추가합니다.
    """
    home_file = config.get('home_file', 'Home.md')
    base_url = config['base_url']

    page_by_filename = {p.filename: p for p in pages}
    page_by_slug = {p.slug: p for p in pages}

    home_page = page_by_filename.get(home_file)
    if home_page is None:
        _warn(
            f"홈 파일 '{home_file}'을 찾을 수 없습니다. "
            f"파일들을 알파벳순으로 처리합니다."
        )
        return sorted(pages, key=lambda p: p.filename.lower())

    link_re = re.compile(
        r'\[([^\]]*)\]\(' + re.escape(base_url) + r'/([^)#\s]+)(?:#[^)]*)?\)'
    )
    ordered_slugs = []
    seen_slugs = set()
    for match in link_re.finditer(home_page.raw_content):
        slug = match.group(2)
        if slug not in seen_slugs:
            ordered_slugs.append(slug)
            seen_slugs.add(slug)

    ordered: List[WikiPage] = [home_page]
    referenced_filenames = {home_page.filename}

    for slug in ordered_slugs:
        page = page_by_slug.get(slug)
        if page is None:
            _warn(f"Home.md의 링크 슬러그 '{slug}'에 해당하는 .md 파일이 없습니다.")
            continue
        if page.filename not in referenced_filenames:
            ordered.append(page)
            referenced_filenames.add(page.filename)

    orphans = [p for p in sorted(pages, key=lambda p: p.filename.lower())
               if p.filename not in referenced_filenames]
    for orphan in orphans:
        _warn(
            f"{orphan.filename} — Home.md 내비게이션에 없습니다. "
            f"알파벳순으로 뒤에 추가합니다."
        )
        ordered.append(orphan)

    if verbose:
        print("\n파일 순서 (Home.md 기반 자동 감지):")
        for i, page in enumerate(ordered, 1):
            print(f"  {i:2d}. {page.filename}")
        print()

    return ordered


def resolve_explicit_order(
    filenames: List[str],
    pages: List[WikiPage],
    wiki_dir: Path,
) -> List[WikiPage]:
    """명시적으로 지정된 파일 순서대로 WikiPage 목록을 반환합니다."""
    page_by_filename = {p.filename: p for p in pages}
    ordered = []
    for filename in filenames:
        if not filename.endswith('.md'):
            filename = filename + '.md'
        page = page_by_filename.get(filename)
        if page is None:
            _warn(f"files 목록의 '{filename}'을 wiki 디렉토리에서 찾을 수 없습니다.")
            continue
        ordered.append(page)

    listed = {p.filename for p in ordered}
    for page in sorted(pages, key=lambda p: p.filename.lower()):
        if page.filename not in listed:
            _warn(f"{page.filename} — files 목록에 없어 병합에서 제외됩니다.")

    return ordered


# ---------------------------------------------------------------------------
# Per-File Content Processing
# ---------------------------------------------------------------------------

def bump_heading_levels(content: str, filename: str) -> str:
    """
    모든 헤딩 레벨을 1 증가시킵니다. (# → ##, ## → ###, ..., 최대 ######)
    코드 블록(```) 내부의 헤딩은 건드리지 않습니다.
    """
    lines = content.splitlines(keepends=True)
    result = []
    in_code_block = False

    for line in lines:
        if re.match(r'^\s*```', line):
            in_code_block = not in_code_block

        if not in_code_block and re.match(r'^\s{0,3}#{1,6}\s', line):
            level = len(re.match(r'^\s{0,3}(#+)', line).group(1))
            if level >= 6:
                _warn(f"{filename} — 헤딩 레벨 6이 bump 범위를 초과합니다: {line.strip()!r}")
                result.append(line)
            else:
                result.append('#' + line)
        else:
            result.append(line)

    return ''.join(result)


def convert_links(
    content: str,
    base_url: str,
    slug_anchor_map: Dict[str, str],
    source_filename: str,
    verbose: bool = False,
) -> str:
    """
    내부 Wiki 링크를 문서 내부 앵커로 변환합니다.

    - base_url/Slug        → #slug_anchor_map[Slug]
    - base_url/Slug#sec    → #sec
    - 그 외 URL            → 변경 없음
    """
    prefix = base_url.rstrip('/') + '/'
    link_re = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')

    warnings_emitted = []

    def replace(m: re.Match) -> str:
        text = m.group(1)
        url = m.group(2)

        if not url.startswith(prefix):
            return m.group(0)

        remainder = url[len(prefix):]

        if '#' in remainder:
            _, section_anchor = remainder.split('#', 1)
            new_url = '#' + section_anchor
        else:
            slug = remainder
            if slug in slug_anchor_map:
                new_url = '#' + slug_anchor_map[slug]
            else:
                warnings_emitted.append(
                    f"{source_filename} — 해석 불가 Wiki 링크: [{text}]({url})\n"
                    f"  슬러그 '{slug}'에 해당하는 파일이 없습니다. 링크를 그대로 유지합니다."
                )
                return m.group(0)

        if verbose:
            print(f"  변환: [{text}]({url})\n"
                  f"      → [{text}]({new_url})")
        return f'[{text}]({new_url})'

    result = link_re.sub(replace, content)

    for warning in warnings_emitted:
        _warn(warning)

    return result


def process_page(
    page: WikiPage,
    config: dict,
    slug_anchor_map: Dict[str, str],
    verbose: bool = False,
) -> str:
    """단일 파일에 inject_heading / bump_headings / convert_links 를 적용합니다."""
    per_file_cfg = (config.get('per_file', {}) or {}).get(page.filename, {}) or {}
    content = page.raw_content

    inject = per_file_cfg.get('inject_heading', None)
    if inject:
        content = inject.strip() + '\n\n' + content

    bump = per_file_cfg.get('bump_headings', config.get('bump_headings', False))
    if bump:
        content = bump_heading_levels(content, page.filename)

    content = convert_links(
        content,
        config['base_url'],
        slug_anchor_map,
        page.filename,
        verbose=verbose,
    )

    return content


# ---------------------------------------------------------------------------
# Cover Page
# ---------------------------------------------------------------------------

def _build_cover_block(config: dict, wiki_dir: Path) -> str:
    """
    wikicom.yaml의 cover 설정으로 HTML cover block 문자열을 생성합니다.
    cover 설정이 없으면 빈 문자열을 반환합니다.

    생성된 블록은 <div class="cover-page">...</div> 형태로,
    merged .md 맨 앞에 삽입되며 Python-Markdown extra 확장의
    HTML passthrough로 그대로 렌더됩니다.
    """
    cover = config.get('cover')
    if not cover:
        return ''

    title    = cover.get('title') or re.sub(r'\.wiki$', '', wiki_dir.name)
    subtitle = cover.get('subtitle', '')
    author   = cover.get('author', '')
    version  = cover.get('version', '')
    date     = cover.get('date', '')

    lines = [f'\n\n# {title}\n']
    if subtitle:
        lines.append(f'\n**{subtitle}**\n')
    meta = ' &nbsp;|&nbsp; '.join(filter(None, [author, version, date]))
    if meta:
        lines.append(f'\n{meta}\n')
        
    return ''.join(lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(
    ordered_pages: List[WikiPage],
    config: dict,
    slug_anchor_map: Dict[str, str],
    verbose: bool = False,
) -> str:
    """모든 페이지를 처리하고 page_break로 연결하여 단일 문자열을 반환합니다."""
    page_break = config['page_break']
    parts = []

    for page in ordered_pages:
        if verbose:
            print(f"처리 중: {page.filename}  (앵커: #{page.anchor})")
        processed = process_page(page, config, slug_anchor_map, verbose=verbose)
        parts.append(processed.rstrip())

    return page_break.join(parts) + '\n'


# ---------------------------------------------------------------------------
# Clone Command
# ---------------------------------------------------------------------------

def parse_wiki_url(url: str) -> dict:
    """
    GitHub wiki clone URL을 파싱하여 구성 요소를 반환합니다.

    입력:  "https://github.com/user/repo.wiki.git"
    반환:
        {
            "base_url":       "https://github.com/user/repo/wiki",
            "clone_url":      "https://github.com/user/repo.wiki.git",
            "repo_wiki_name": "repo.wiki",
            "user":           "user",
            "repo":           "repo",
            "internal":       "./wikispace/user.repo.wiki",
        }
    """
    m = re.match(r'^https://github\.com/([^/]+)/([^/]+)\.wiki\.git$', url)
    if not m:
        _error(
            f"올바른 GitHub wiki clone URL이 아닙니다: {url}\n"
            "  형식: https://github.com/유저/리포.wiki.git"
        )
    user, repo = m.group(1), m.group(2)
    return {
        "base_url":       f"https://github.com/{user}/{repo}/wiki",
        "clone_url":      url,
        "repo_wiki_name": f"{user}.{repo}.wiki",
        "user":           user,
        "repo":           repo,
    }


def check_git_available() -> None:
    """git이 PATH에 있는지 확인합니다. 없으면 에러 후 종료."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        _error(
            "git이 설치되어 있지 않거나 PATH에 없습니다.\n"
            "  설치: https://git-scm.com/downloads"
        )


def run_git_clone(clone_url: str, target_dir: Path) -> None:
    """git clone을 실행합니다. 실패 시 에러 후 종료."""
    result = subprocess.run(
        ["git", "clone", clone_url, str(target_dir)],
    )
    if result.returncode != 0:
        _error(f"git clone 실패 (exit code {result.returncode})")


def write_wikicom_yaml(yaml_path: Path, base_url: str) -> None:
    """wikicom.yaml을 주석 포함 형태로 생성합니다."""
    content = f"""\
# wikicom.yaml — wikicom clone으로 자동 생성된 설정 파일
# 필요에 따라 수정하세요.

# 필수: GitHub Wiki 기준 URL (내부 링크 변환에 사용)
base_url: "{base_url}"

# 출력 파일 경로 (기본: mergespace/{{wiki-folder-name}}.md)
# output: "mergespace/repo.wiki.md"

# 파일 간 구분자 (PDF용: "\\\\newpage")
page_break: "\\n\\n---\\n\\n"

# 파일 순서 자동 감지에 사용할 홈 파일
home_file: "Home.md"

# 헤딩 레벨 일괄 증가 (# → ##, ## → ### ...)
bump_headings: false

# 병합에서 제외할 파일 목록
exclude: []

# 파일별 세부 설정
# per_file:
#   Page-Name.md:
#     inject_heading: "# Page Title"   # 레벨-1 헤딩이 없는 파일에 삽입
#     bump_headings: true              # 이 파일만 헤딩 레벨 증가

# 표지 설정 (없으면 표지 미생성)
# cover:
#   title: "My Project"              # 없으면 wiki 폴더명에서 자동 추출
#   subtitle: "v1.0 Documentation"
#   author: "Author Name"
#   version: "1.0.0"
#   date: "2026-02-28"
"""
    yaml_path.write_text(content, encoding='utf-8')


def write_wiki_gitignore(wiki_dir: Path) -> None:
    """wiki 클론 폴더 내부에 .gitignore를 생성합니다.

    git pull/fetch 시 wikicom이 생성한 파일(merged.md, wikicom.yaml)이
    untracked 파일로 감지되지 않도록 wiki 레포 자체의 .gitignore에 등록합니다.
    """
    gitignore_path = wiki_dir / '.gitignore'
    content = "# wikicom 생성 파일\nwikicom.yaml\n"

    existing = gitignore_path.read_text(encoding='utf-8') if gitignore_path.exists() else ''

    entries_to_add = [
        line for line in content.splitlines(keepends=True)
        if not line.startswith('#') and line.strip() and line not in existing
    ]
    if not entries_to_add:
        return

    try:
        with open(gitignore_path, 'a', encoding='utf-8') as f:
            if existing and not existing.endswith('\n'):
                f.write('\n')
            f.write("# wikicom 생성 파일\n")
            f.writelines(entries_to_add)
    except OSError as e:
        _warn(f"wiki .gitignore 생성 실패: {e}")


def cmd_clone(args) -> None:
    """wikicom clone <url> [--dir DIR] 실행."""
    check_git_available()

    url_info = parse_wiki_url(args.url)

    wikispace_dir = Path(args.dir).resolve() if args.dir else Path('wikispace').resolve()    
    target_dir = wikispace_dir / url_info['repo_wiki_name']

    if target_dir.exists() and any(target_dir.iterdir()):
        _error(
            f"대상 디렉토리가 이미 존재합니다: {target_dir}\n"
            "  삭제하거나 --dir 옵션으로 다른 위치를 지정하세요."
        )

    wikispace_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {url_info['clone_url']} ...")
    run_git_clone(url_info['clone_url'], target_dir)

    yaml_path = target_dir / 'wikicom.yaml'
    write_wikicom_yaml(yaml_path, url_info['base_url'])

    write_wiki_gitignore(target_dir)

    print(f"\nCloned:    {target_dir}")
    print(f"Config:    {yaml_path}")
    print(f"Gitignore: {target_dir / '.gitignore'}  (merged.md, wikicom.yaml 제외)")
    print(f"\nNext step:")
    print(f"  wikicom {target_dir}")


# ---------------------------------------------------------------------------
# Pull Command
# ---------------------------------------------------------------------------

def get_current_branch(wiki_dir: Path) -> str:
    """현재 브랜치명을 반환합니다. 실패 시 에러 후 종료."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(wiki_dir),
    )
    if result.returncode != 0:
        _error(f"브랜치 감지 실패: {result.stderr.strip()}")
    return result.stdout.strip()


def cmd_pull(args) -> None:
    """wikicom pull <wiki-dir> [--force] 실행."""
    check_git_available()

    wiki_dir = Path(args.wiki_dir).resolve()
    if not wiki_dir.is_dir():
        _error(f"디렉토리가 존재하지 않습니다: {wiki_dir}")
    if not (wiki_dir / '.git').exists():
        _error(f"git 레포지터리가 아닙니다: {wiki_dir}")

    if args.force:
        print(f"강제 동기화: {wiki_dir}")
        print("  git fetch origin ...")
        fetch = subprocess.run(
            ["git", "fetch", "origin"],
            cwd=str(wiki_dir),
        )
        if fetch.returncode != 0:
            _error(f"git fetch 실패 (exit code {fetch.returncode})")

        branch = get_current_branch(wiki_dir)
        print(f"  git reset --hard origin/{branch} ...")
        reset = subprocess.run(
            ["git", "reset", "--hard", f"origin/{branch}"],
            cwd=str(wiki_dir),
        )
        if reset.returncode != 0:
            _error(f"git reset --hard 실패 (exit code {reset.returncode})")

        print(f"\n완료: origin/{branch} 으로 강제 동기화됨")
    else:
        print(f"Pull: {wiki_dir}")
        pull = subprocess.run(
            ["git", "pull"],
            cwd=str(wiki_dir),
        )
        if pull.returncode != 0:
            _error(
                f"git pull 실패 (exit code {pull.returncode})\n"
                "  충돌이 발생했다면 --force 옵션으로 원격 상태로 강제 동기화할 수 있습니다."
            )
        print("\n완료: 최신 상태로 업데이트됨")


# ---------------------------------------------------------------------------
# Render Command
# ---------------------------------------------------------------------------

def _ensure_heading_ids(body: str) -> str:
    """
    id 없는 헤딩 요소에 heading_to_anchor 기준 id를 삽입합니다.

    toc 확장 미사용(--no-toc) 시에도 앵커 링크 타깃이 유효하도록 보완합니다.
    이미 id가 있는 헤딩은 건드리지 않습니다.
    중복 헤딩 텍스트는 -1, -2, ... 접미사로 구분합니다.
    """
    seen: Dict[str, int] = {}

    def add_id(m: re.Match) -> str:
        tag = m.group(1)    # 'h1', 'h2', ...
        attrs = m.group(2)  # 기존 속성 문자열 (공백 포함, 없으면 '')
        inner = m.group(3)  # 내부 HTML

        if 'id=' in attrs:
            return m.group(0)

        plain = re.sub(r'<[^>]+>', '', inner)
        base = heading_to_anchor(plain)
        if not base:
            return m.group(0)

        count = seen.get(base, 0)
        anchor = base if count == 0 else f'{base}-{count}'
        seen[base] = count + 1

        return f'<{tag}{attrs} id="{anchor}">{inner}</{tag}>'

    return re.sub(r'<(h[1-6])([^>]*)>(.*?)</\1>', add_id, body, flags=re.DOTALL)


def render_md_to_html(
    md_path: Path,
    output_path: Path,
    title: Optional[str],
    css: str,
    include_toc: bool,
) -> None:
    """단일 .md 파일을 HTML로 변환하여 output_path에 저장합니다."""
    import markdown as md_lib  # lazy import (선택적 의존성)

    text = md_path.read_text(encoding='utf-8')
    # Python-Markdown은 CommonMark와 달리 1~3칸 들여쓰기 ATX 헤딩을 인식하지 못합니다.
    # 들여쓰기를 제거하여 헤딩이 올바르게 변환되도록 정규화합니다.
    text = re.sub(r'^( {1,3})(#{1,6}(?:[ \t]|$))', r'\2', text, flags=re.MULTILINE)
    extensions = ['extra', 'toc', 'attr_list'] if include_toc else ['extra']
    md = md_lib.Markdown(
        extensions=extensions,
        extension_configs={'toc': {'toc_depth': '1-3'}},
    )
    body = md.convert(text)
    body = _ensure_heading_ids(body)

    # 두 번째 <h1>부터 앞에 page-break div 삽입 (첫 번째는 건너뜀)
    _first = [True]

    def _insert_page_break(m: re.Match) -> str:
        if _first:
            _first.pop()
            return m.group(0)
        return '<div class="page-break"></div>\n' + m.group(0)

    body = re.sub(r'<h1(?=[\s>])', _insert_page_break, body)

    toc_html = ''
    if include_toc and '<li>' in getattr(md, 'toc', ''):
        toc_html = f'<nav class="toc">\n{md.toc}\n</nav>\n'

    if title is None:
        title = md_path.stem

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
{toc_html}{body}
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')


def cmd_render(args) -> None:
    """wikicom render [INPUT] [options] 실행."""
    try:
        import markdown as _  # noqa: F401
    except ImportError:
        _error("markdown 패키지가 설치되어 있지 않습니다. 'pip install markdown' 실행 후 재시도하세요.")

    input_path = Path(args.input).resolve() if args.input else Path.cwd() / 'mergespace'

    css = DEFAULT_CSS
    if args.css:
        css_file = Path(args.css)
        if not css_file.exists():
            _error(f"CSS 파일을 찾을 수 없습니다: {css_file}")
        css = css_file.read_text(encoding='utf-8')

    include_toc = not getattr(args, 'no_toc', False)
    title = getattr(args, 'title', None)

    if input_path.is_file():
        if input_path.suffix.lower() != '.md':
            _error(f"입력 파일이 .md 형식이 아닙니다: {input_path}")
        output_path = Path(args.output) if args.output else (
            Path.cwd() / 'htmlspace' / input_path.with_suffix('.html').name
        )
        render_md_to_html(input_path, output_path, title, css, include_toc)
        print(f"완료: {input_path.name} → {output_path}")

    elif input_path.is_dir():
        md_files = sorted(input_path.glob('*.md'))
        if not md_files:
            _error(f"{input_path} 에서 .md 파일을 찾을 수 없습니다.")
        output_dir = Path(args.output) if args.output else Path.cwd() / 'htmlspace'
        for md_file in md_files:
            out = output_dir / md_file.with_suffix('.html').name
            render_md_to_html(md_file, out, title or md_file.stem, css, include_toc)
        print(f"완료: {len(md_files)}개 파일 → {output_dir}")

    else:
        _error(f"경로가 존재하지 않습니다: {input_path}")


# ---------------------------------------------------------------------------
# PDF Command
# ---------------------------------------------------------------------------

def _html_to_pdf(html_path: Path, output_path: Path, fmt: str, margins: dict) -> None:
    """Playwright Chromium으로 로컬 HTML 파일을 PDF로 변환합니다."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _error(
            "playwright가 설치되어 있지 않습니다.\n"
            "  pip install playwright && playwright install chromium"
        )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(output_path),
            format=fmt,
            print_background=True,
            prefer_css_page_size=True,
            margin=margins,
        )
        browser.close()


def cmd_pdf(args) -> None:
    """wikicom pdf [INPUT] [options] 실행."""
    input_path = Path(args.input).resolve() if args.input else Path.cwd() / 'htmlspace'
    fmt = args.format
    margins = {"top": "20px", "bottom": "20px", "left": "20px", "right": "20px"}

    if input_path.is_file():
        if input_path.suffix.lower() != '.html':
            _error(f"입력 파일이 .html 형식이 아닙니다: {input_path}")
        output_path = Path(args.output) if args.output else (
            Path.cwd() / 'pdfspace' / input_path.with_suffix('.pdf').name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _html_to_pdf(input_path, output_path, fmt, margins)
        print(f"완료: {input_path.name} → {output_path}")

    elif input_path.is_dir():
        html_files = sorted(input_path.glob('*.html'))
        if not html_files:
            _error(f"{input_path} 에서 .html 파일을 찾을 수 없습니다.")
        output_dir = Path(args.output) if args.output else Path.cwd() / 'pdfspace'
        output_dir.mkdir(parents=True, exist_ok=True)
        for html_file in html_files:
            out = output_dir / html_file.with_suffix('.pdf').name
            _html_to_pdf(html_file, out, fmt, margins)
        print(f"완료: {len(html_files)}개 파일 → {output_dir}")

    else:
        _error(f"경로가 존재하지 않습니다: {input_path}")


# ---------------------------------------------------------------------------
# Merge Command
# ---------------------------------------------------------------------------

def cmd_merge(args) -> None:
    """wikicom [merge] <wiki-dir> [options] 실행."""
    wiki_dir = Path(args.wiki_dir).resolve()
    if not wiki_dir.is_dir():
        _error(f"디렉토리가 존재하지 않습니다: {wiki_dir}")

    config = load_config(wiki_dir, args)
    verbose = getattr(args, 'verbose', False)

    if verbose:
        print(f"Wiki 디렉토리: {wiki_dir}")
        print(f"Base URL:      {config['base_url']}")
        print(f"출력 파일:     {config['output']}")
        print(f"페이지 구분자: {config['page_break']!r}")
        print(f"헤딩 bump:     {config['bump_headings']}")
        print()

    pages = discover_files(wiki_dir, config)
    if not pages:
        _error(f"{wiki_dir} 에서 .md 파일을 찾을 수 없습니다.")

    slug_anchor_map = build_slug_anchor_map(pages)

    if verbose:
        print("슬러그 → 앵커 매핑:")
        for slug, anchor in sorted(slug_anchor_map.items()):
            print(f"  {slug} → #{anchor}")
        print()

    explicit_files = config.get('files')
    if explicit_files:
        ordered = resolve_explicit_order(explicit_files, pages, wiki_dir)
    elif config.get('no_auto_order'):
        ordered = sorted(pages, key=lambda p: p.filename.lower())
        if verbose:
            print("파일 순서 (알파벳순):")
            for i, p in enumerate(ordered, 1):
                print(f"  {i:2d}. {p.filename}")
            print()
    else:
        ordered = infer_order_from_home(
            pages, slug_anchor_map, config, wiki_dir, verbose=verbose
        )

    if not ordered:
        _error("병합할 파일이 없습니다.")

    merged = assemble(ordered, config, slug_anchor_map, verbose=verbose)

    cover = _build_cover_block(config, wiki_dir)
    if cover:
        merged = cover + '\n\n' + merged

    if args.dry_run:
        sys.stdout.buffer.write(merged.encode('utf-8'))
    else:
        output_path = Path(config['output'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(merged, encoding='utf-8')
        print(f"\n완료: {len(ordered)}개 파일 → {output_path}")

    if _warnings:
        print(f"\n발생한 경고 ({len(_warnings)}건):", file=sys.stderr)
        for w in _warnings:
            print(f"  · {w.replace('WARNING: ', '')}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI — Subparser Builders
# ---------------------------------------------------------------------------


def _add_clone_parser(subparsers) -> None:
    p = subparsers.add_parser(
        'clone',
        help='GitHub wiki를 클론하고 wikicom.yaml을 자동 생성합니다',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom clone https://github.com/user/repo.wiki.git\n"
            "  wikicom clone https://github.com/user/repo.wiki.git --dir ./myspace\n"
        ),
    )
    p.add_argument(
        'url',
        metavar='URL',
        help='GitHub wiki clone URL (예: https://github.com/user/repo.wiki.git)',
    )
    p.add_argument(
        '--dir',
        metavar='DIR',
        default=None,
        help='wikispace 부모 디렉토리 (기본: ./wikispace)',
    )
    p.set_defaults(func=cmd_clone)


def _add_pull_parser(subparsers) -> None:
    p = subparsers.add_parser(
        'pull',
        help='wiki를 원격 최신 상태로 업데이트합니다',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom pull wikispace/repo.wiki\n"
            "  wikicom pull wikispace/repo.wiki --force\n"
        ),
    )
    p.add_argument(
        'wiki_dir',
        metavar='WIKI_DIR',
        help='업데이트할 wiki 클론 디렉토리',
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='충돌 무시 후 원격 상태로 강제 동기화 (git fetch + reset --hard)',
    )
    p.set_defaults(func=cmd_pull)


def _add_merge_parser(subparsers) -> None:
    p = subparsers.add_parser(
        'merge',
        help='wiki .md 파일들을 단일 문서로 병합합니다',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom merge wikispace/repo.wiki\n"
            "  wikicom merge wikispace/repo.wiki --output out.md\n"
            "  wikicom merge wikispace/repo.wiki --bump-headings --dry-run\n"
        ),
    )
    p.add_argument(
        'wiki_dir',
        metavar='WIKI_DIR',
        help='.md 파일이 있는 wiki 클론 디렉토리',
    )
    p.add_argument(
        '--base-url',
        metavar='URL',
        help='Wiki 기준 URL (wikicom.yaml이 없을 때 필요)',
    )
    p.add_argument(
        '--output',
        metavar='FILE',
        help='출력 .md 파일 경로 (기본: mergespace/{wiki-folder-name}.md)',
    )
    p.add_argument(
        '--config',
        metavar='FILE',
        help='wikicom.yaml 경로 명시',
    )
    p.add_argument(
        '--page-break',
        metavar='STR',
        help=r'파일 간 구분자 (기본: \n\n---\n\n, PDF용: \\newpage)',
    )
    p.add_argument(
        '--bump-headings',
        action='store_true',
        help='헤딩 레벨을 1 증가 (# → ##, ## → ###, ...)',
    )
    p.add_argument(
        '--home-file',
        metavar='FILE',
        help='파일 순서 자동 감지용 홈 파일 (기본: Home.md)',
    )
    p.add_argument(
        '--no-auto-order',
        action='store_true',
        help='Home.md 기반 자동 순서 감지 비활성화 (알파벳순)',
    )
    p.add_argument(
        '--verbose',
        action='store_true',
        help='링크 변환 내역 및 처리 상세 출력',
    )
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='파일 쓰기 없이 결과를 stdout에 출력',
    )
    p.set_defaults(func=cmd_merge)


def _add_render_parser(subparsers) -> None:
    p = subparsers.add_parser(
        'render',
        help='.md 파일을 HTML로 변환합니다',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom render\n"
            "  wikicom render mergespace/repo.wiki.md\n"
            "  wikicom render mergespace/ --output htmlspace/\n"
            "  wikicom render input.md --title 'My Doc' --no-toc\n"
        ),
    )
    p.add_argument(
        'input',
        metavar='INPUT',
        nargs='?',
        default=None,
        help='변환할 .md 파일 또는 디렉토리 (기본: ./mergespace/)',
    )
    p.add_argument(
        '--output',
        metavar='PATH',
        help='출력 .html 파일 또는 디렉토리 경로 (기본: ./htmlspace/)',
    )
    p.add_argument(
        '--title',
        metavar='TITLE',
        help='HTML <title> 태그 내용 (기본: 파일명)',
    )
    p.add_argument(
        '--css',
        metavar='FILE',
        help='적용할 .css 파일 경로 (기본: 내장 GitHub 스타일)',
    )
    p.add_argument(
        '--no-toc',
        action='store_true',
        help='목차(TOC) 생성 비활성화',
    )
    p.set_defaults(func=cmd_render)


def _add_pdf_parser(subparsers) -> None:
    p = subparsers.add_parser(
        'pdf',
        help='HTML을 PDF로 변환합니다 (Playwright 필요)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom pdf\n"
            "  wikicom pdf htmlspace/repo.wiki.html\n"
            "  wikicom pdf htmlspace/ --output pdfspace/\n"
        ),
    )
    p.add_argument(
        'input',
        metavar='INPUT',
        nargs='?',
        default=None,
        help='변환할 .html 파일 또는 디렉토리 (기본: ./htmlspace/)',
    )
    p.add_argument(
        '--output',
        metavar='PATH',
        help='출력 .pdf 파일 또는 디렉토리 경로 (기본: ./pdfspace/)',
    )
    p.add_argument(
        '--format',
        metavar='FORMAT',
        default='A4',
        help='페이지 크기 (기본: A4, 예: Letter, A3)',
    )
    p.set_defaults(func=cmd_pdf)


# ---------------------------------------------------------------------------
# CLI — Entry Point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='wikicom',
        description='GitHub Wiki 클로너 & Markdown 병합기.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom clone https://github.com/user/repo.wiki.git\n"
            "  wikicom clone https://github.com/user/repo.wiki.git --dir ./docs\n"
            "  wikicom pull wikispace/repo.wiki\n"
            "  wikicom pull wikispace/repo.wiki --force\n"
            "  wikicom wikispace/repo.wiki\n"
            "  wikicom merge wikispace/repo.wiki --output out.md\n"
            "  wikicom merge wikispace/repo.wiki --bump-headings --dry-run\n"
            "  wikicom render\n"
            "  wikicom render mergespace/repo.wiki.md\n"
        ),
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'wikicom {VERSION}',
    )

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    _add_clone_parser(subparsers)
    _add_pull_parser(subparsers)
    _add_merge_parser(subparsers)
    _add_render_parser(subparsers)
    _add_pdf_parser(subparsers)

    return parser


def main() -> None:
    argv = sys.argv[1:]

    # 첫 번째 positional이 서브커맨드가 아니면 'merge'를 암묵적으로 삽입
    first_positional = next((a for a in argv if not a.startswith('-')), None)
    if first_positional is not None and first_positional not in SUBCOMMANDS:
        argv = ['merge'] + argv

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
