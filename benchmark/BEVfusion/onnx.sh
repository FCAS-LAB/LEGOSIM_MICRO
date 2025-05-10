app=${1}

mkdir ${app}
cd ${app}
/home/qc/onnx2c/build/onnx2c ../swint/${app}.onnx > ${app}.c