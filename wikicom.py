#!/usr/bin/env python3
"""
wikicom — GitHub Wiki 클로너 & Markdown 병합기

GitHub Wiki를 자동으로 클론하고, 여러 .md 파일을 단일 .md 파일로 통합합니다.
내부 Wiki 링크를 문서 내부 앵커(#anchor)로 자동 변환합니다.

Usage:
    wikicom clone <url> [--dir DIR]
    wikicom pull <wiki-dir> [--force]
    wikicom [merge] <wiki-dir> [options]

Examples:
    wikicom clone https://github.com/user/repo.wiki.git
    wikicom clone https://github.com/user/repo.wiki.git --dir ./docs
    wikicom pull wikispace/repo.wiki
    wikicom pull wikispace/repo.wiki --force
    wikicom wikispace/repo.wiki
    wikicom merge wikispace/repo.wiki --output merged.md
    wikicom merge wikispace/repo.wiki --bump-headings --dry-run

Dependencies:
    pip install pyyaml
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

VERSION = "1.0.0"


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
    if getattr(cli_args, 'home', None):
        cfg['home_file'] = cli_args.home
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
        "repo_wiki_name": f"{repo}.wiki",
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
    elif getattr(args, 'no_auto_order', False) or config.get('no_auto_order'):
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
# CLI
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
        ),
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'wikicom {VERSION}',
    )

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    # --- clone ---
    clone_p = subparsers.add_parser(
        'clone',
        help='GitHub wiki를 클론하고 wikicom.yaml을 자동 생성합니다',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom clone https://github.com/user/repo.wiki.git\n"
            "  wikicom clone https://github.com/user/repo.wiki.git --dir ./myspace\n"
        ),
    )
    clone_p.add_argument(
        'url',
        metavar='URL',
        help='GitHub wiki clone URL (예: https://github.com/user/repo.wiki.git)',
    )
    clone_p.add_argument(
        '--dir',
        metavar='DIR',
        default=None,
        help='wikispace 부모 디렉토리 (기본: ./wikispace)',
    )
    clone_p.set_defaults(func=cmd_clone)

    # --- pull ---
    pull_p = subparsers.add_parser(
        'pull',
        help='wiki를 원격 최신 상태로 업데이트합니다',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wikicom pull wikispace/repo.wiki\n"
            "  wikicom pull wikispace/repo.wiki --force\n"
        ),
    )
    pull_p.add_argument(
        'wiki_dir',
        metavar='WIKI_DIR',
        help='업데이트할 wiki 클론 디렉토리',
    )
    pull_p.add_argument(
        '--force', '-f',
        action='store_true',
        help='충돌 무시 후 원격 상태로 강제 동기화 (git fetch + reset --hard)',
    )
    pull_p.set_defaults(func=cmd_pull)

    # --- merge ---
    merge_p = subparsers.add_parser(
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
    merge_p.add_argument(
        'wiki_dir',
        metavar='WIKI_DIR',
        help='.md 파일이 있는 wiki 클론 디렉토리',
    )
    merge_p.add_argument(
        '--base-url',
        metavar='URL',
        help='Wiki 기준 URL (wikicom.yaml이 없을 때 필요)',
    )
    merge_p.add_argument(
        '--output', '-o',
        metavar='FILE',
        help='출력 .md 파일 경로 (기본: mergespace/{wiki-folder-name}.md)',
    )
    merge_p.add_argument(
        '--config',
        metavar='FILE',
        help='wikicom.yaml 경로 명시',
    )
    merge_p.add_argument(
        '--page-break',
        metavar='STR',
        help=r'파일 간 구분자 (기본: \n\n---\n\n, PDF용: \\newpage)',
    )
    merge_p.add_argument(
        '--bump-headings',
        action='store_true',
        help='헤딩 레벨을 1 증가 (# → ##, ## → ###, ...)',
    )
    merge_p.add_argument(
        '--home',
        metavar='FILE',
        help='파일 순서 자동 감지용 홈 파일 (기본: Home.md)',
    )
    merge_p.add_argument(
        '--no-auto-order',
        action='store_true',
        help='Home.md 기반 자동 순서 감지 비활성화 (알파벳순)',
    )
    merge_p.add_argument(
        '--verbose',
        action='store_true',
        help='링크 변환 내역 및 처리 상세 출력',
    )
    merge_p.add_argument(
        '--dry-run',
        action='store_true',
        help='파일 쓰기 없이 결과를 stdout에 출력',
    )
    merge_p.set_defaults(func=cmd_merge)

    return parser


def main() -> None:
    SUBCOMMANDS = {'clone', 'merge', 'pull'}

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
