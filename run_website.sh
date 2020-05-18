#!/bin/bash
loginfo() { echo "[INFO] $@"; }
logerror() { echo "[ERROR] $@" 1>&2; }

python3 src/script.py "home" "book"
rm -rf node_modules/gitbook-plugin-tbfed-pagefooter
gitbook install
python3 src/script.py "home" "powered"
python3 src/script.py "home" "gitalk"
gitbook build ./ _book

# # rm -rf /opt/apache-tomcat-9.0.17/webapps/test_book
# # cp -r _book /opt/apache-tomcat-9.0.17/webapps/test_book
