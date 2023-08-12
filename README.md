# cv_categorized
Its a machine learning model to automatically  categorize resumes based on their domain (e.g., sales, marketing, etc.)

# Model Choose
Sequential models in Keras are well-suited for building feedforward neural networks and simple sequential architectures where each layer has one input tensor and one output tensor.

# preprocessing or feature extraction
For any Language Language processing model we need a clean corpus. So, I have get read of any punctuation mark, numerical value or extra spaces. Then I saw that the corpus has some null values also. Then I have removed any stop words like as 'I','me', etc using NLTK. Also I have get rid of stem words.

# How to use
In the cv_categorized directory, open a cmd or terminal.
then run this command 
```
pip install -r requirement.txt
```

it could take some because of installing libraries like tensorflow,nltk,pandas,numpy, etc

then run this command 

```
python script.py --dir [path/to/dir]
```

thats it!!
you can find the result in the directory called 'CV_folder' in the same directory. And you will find the 'resume_categorized.csv' file in the same directory.
