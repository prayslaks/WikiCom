# 1단계: 가상환경 활성화
if(-not (Test-Path .venv -PathType Container)) {
    Write-Host "가상환경 생성.."
    python -m venv .venv
    Write-Host "가상환경 생성 완료"
}
Write-Host "가상환경 활성화.."
.\.venv\Scripts\activate
Write-Host "가상환경 활성화 완료"

# 2단계: 필수 패키지 설치
Write-Host "필수 패키지 설치.."
pip install -r requirements.txt
Write-Host "필수 패키지 설치 완료"

# 3단계: 산출물 폴더 생성
$folderList = @("wikispace", "mergespace", "htmlspace", "pdfspace")
$isEmpty = $false
foreach ($folder in $folderList) {
    if (-not (Test-Path .\$folder -PathType Container)) {
        $isEmpty = $true
    }
}
if($isEmpty) {
    Write-Host "산출물 폴더 생성.."
    foreach ($folder in $folderList) {
        if (-not (Test-Path .\$folder -PathType Container)) {
            New-Item -Path .\$folder -ItemType Container
        }
    }
    Write-Host "산출물 폴더 생성 완료"
}

Write-Host "작업 환경 구축 완료"