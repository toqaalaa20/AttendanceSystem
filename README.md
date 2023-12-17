# AttendanceSystem
Real-time face recognition based attendance system using OpenCV, Python.


## How to use
1. Clone the repository

```bash
git clone https://github.com/toqaalaa20/AttendanceSystem
```
2. Install the requirements using
```bash
pip install -r requirements.txt
```

3. Add videos for your desired crowd in the `Videos` folder

4. Run `python generate_dataset.py` to generate dataset for the videos and train the model.

5. Run `python app.py` to start the flask app and visit `localhost:5000` to see the app in action.