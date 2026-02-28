$wikiName = Read-Host "깃 허브 위키 리포지토리를 입력하세요"

if(-not (Test-Path .\wikispace\$wikiName.wiki -PathType Container)) {
    Write-Host "해당 위키는 존재하지 않아 작업 불가"
    return
}

python .\wikicom.py pull .\wikispace\$wikiName.wiki\
python .\wikicom.py .\wikispace\$wikiName.wiki\
python .\wikicom.py render .\mergespace\$wikiName.wiki.md --no-toc
python .\wikicom.py pdf .\htmlspace\$wikiName.wiki.html
