# face_recognition_indian_celebrities
# Steps to setup

## Step 1: Create virtual environment
pip install virtualenv
virtualenv env

## Step 2: Activate virtual environment

For windows : ```env/Script/activate```

For Linux : ```source env/bin/activate```

## Step 3 : Install required modules

Install modules : ``` pip install -r requirements.txts ```

## Step 4 : Download cascade file ,weights and model and save in folder named extract

```mkdir extract```

```gdown --id 1C5H4xM2nAo0XBe-naUCc-2SKDpjtxXdb```

On Linux machine : ```unzip extract.zip```

If on windows platfrom unzip extract.zip using unzipping software like 7zip.

Delete zip file : ```rm extract.zip```

## Step 5 : Run the project

Run python file : ``` python faces_video.py ```

! [output] (output.gif)
