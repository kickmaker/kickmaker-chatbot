from setuptools import setup

setup(
    name='kickbot',
    version='0.1.0',    
    description='Kickmaker chatbot Project',
    url='https://github.com/kickmaker/kickmaker-chatbot.git',
    author='Baptiste Egnell',
    packages=['kickbot'],
    install_requires=['langchain[all]',
                    'pypdf==4.2.0',
                    'streamlit==1.34.0',
                    'chromadb==0.5.0',
                    'huggingface_hub==0.23.0',
                    'transformers==4.40.2',
                    'diffusers==0.27.2 ',
                    'invisible_watermark==0.2.0',
                    'accelerate==0.30.1 ',
                    'safetensors==0.4.3',
                    'cycler==0.12.1',
                    'peft==0.10.0',
                    'kiwisolver==1.4.5',
                    'onnxruntime-openvino==1.17.1'
                     ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
)
