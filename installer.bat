echo "---uninstall previous configuration---"
python setup.py uninstall
echo "---installation ---"
python setup.py install
echo "---check you installation---"
python setup.py check
pause
