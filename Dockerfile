FROM python:3

WORKDIR /app

RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install sklearn
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install plotly
RUN pip3 install dash
RUN pip3 install dash_core_components
RUN pip3 install dash_html_components
RUN pip3 install dash_bootstrap_components

COPY . .

CMD ["python3", "dashboard.py"]

