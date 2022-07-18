FROM tiangolo/uwsgi-nginx-flask:python3.6
RUN pip install --upgrade pip
RUN mkdir /apps
ADD . /apps/
WORKDIR /apps

RUN pip install -r covid_require.txt

EXPOSE 443
EXPOSE 80
EXPOSE 5000
CMD ["python", "/apps/master_code.py"]
