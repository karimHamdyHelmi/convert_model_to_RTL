# convert_model_to_RTL

python convert_model_to_rtl.py `
>>     --model-module QuantizedMNISTNet.py `
>>     --model-class SmallMNISTNet `
>>     --model-path mnist_model.pth `
>>     --out-dir rtl_build `
>>     --scale 256 `
>>     --emit-wrapper-sv `
>>     --copy-rtl-templates `
>>     --rtl-template-dir "C:\Users\Kimo_\OneDrive - Alexandria University\Desktop\test_2\to_rtl"
