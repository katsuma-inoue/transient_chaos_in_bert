# download wikipedia data
mkdir ../data/wiki
wget -P ../data/wiki/ https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python -m pip install wikiextractor
python -m wikiextractor.WikiExtractor -o ../data/wiki/parts --processes 6 ../data/wiki/enwiki-latest-pages-articles.xml.bz2
