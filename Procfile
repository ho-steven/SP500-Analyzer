web: sh setup.sh && streamlit run app.py
heroku buildpacks:add --index 1 heroku/python
heroku buildpacks:add --index 2 numrut/ta-lib
heroku buildpacks:add --index 2 https://github.com/numrut/heroku-buildpack-python-talib
