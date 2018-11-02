/bin/rm -f model6tissues.pkl model_hippo.pkl
THEANO_FLAGS="device=cpu,floatX=float32" python test_import.py with_convaffine
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi
