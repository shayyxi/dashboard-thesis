import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash
import pandas as pd
import numpy as np
import math

#imports of my files
import data_preprocessing as dp 
import data_visualization as dv
import machine_learning as ml


# Specifying the path of CSV file
filepath = "./dataset/student-mat.csv"

# Checking if the file exists or not
dp.check_file_existence(filepath)

# Reading the file
df = dp.read_file(filepath)
df_unencoded = dp.read_file(filepath)

# Checking for missing data in the dataframe
dp.check_missing_data(df)

# Checking for plausibility problems in the dataframe
dp.plausibility_check(df)

# Fetching the number of rows and columns of dataset
number_of_rows = dp.number_of_rows(df)
number_of_cols = dp.number_of_cols(df)

# Specifying the column names to be used for regression based on our initial research for G3
col_names_regression_g3=["Mjob","Fedu","school","sex","Medu","age","Pstatus","address","famsize","G1","G2"]


# Specifying the column names to be used for regression based on our initial research for G2
col_names_regression_g2=["Fedu","Medu","studytime", "G1"]

# Specifying the column names to be label encoded so they can be used in algorithm
col_names_encoding=["Mjob","school","Pstatus","sex","address","famsize"]

# Label encoding the columns
df = ml.label_encoding(df,col_names_encoding)

# Fitting a random forest model on the dataframe
regressor_g3, r2_score_g3 = ml.random_forest_regressor_g3(df)
regressor_g2, r2_score_g2 = ml.random_forest_regressor_g2(df)

# Details of model
details_of_model_g3 = "The above simulation tool can be used to predict the value of G3. The model we have implemented on our dataset is the random forest regression model. We tried different models as well but we achieved the maximum accuracy with this model. The accuracy of our model is " + str(round(r2_score_g3*100,2)) + "%. So, it is a pretty accurate model. After a lot of trial and error, we came up with this model and the features to use in it. With the sliders, you can try different values and use it to predict the score for students. The chart in top right shows the most significant features of our model which are contributing most in the model.";

details_of_model_g2 = "The above simulation tool can be used to predict the value of G2. The model we have implemented on our dataset is the random forest regression model. We tried different models as well but we achieved the maximum accuracy with this model. The accuracy of our model is " + str(round(r2_score_g2*100,2)) + "%. So, it is a pretty accurate model. After a lot of trial and error, we came up with this model and the features to use in it. With the sliders, you can try different values and use it to predict the score for students. The chart in top right shows the most significant features of our model which are contributing most in the model.";


# Selecting the most significant features from the model for G3 predictor
df_feature_importances_g3 = pd.DataFrame(regressor_g3.feature_importances_*100,columns=["Importance"],index=col_names_regression_g3)
df_feature_importances_g3 = df_feature_importances_g3.sort_values("Importance", ascending=False)


# Selecting the most significant features from the model for G2 predictor
df_feature_importances_g2 = pd.DataFrame(regressor_g2.feature_importances_*100,columns=["Importance"],index=col_names_regression_g2)
df_feature_importances_g2 = df_feature_importances_g2.sort_values("Importance", ascending=False)



fig_features_importance_g3 = dv.feature_importance_chart(df_feature_importances_g3)
fig_features_importance_g2 = dv.feature_importance_chart(df_feature_importances_g2)




fig_correlation_heatmap = dv.correlation_heatmap(df_unencoded)
fig_boxplots_numeric_columns = dv.boxplots_numeric_columns(df)
fig_sex_mjob_school_columns = dv.distribution_sex_mjob_school(df_unencoded)
fig_address_pstatus_medu = dv.distribution_address_pstatus_medu(df_unencoded)
fig_guardian_fjob_fedu = dv.distribution_guardian_fjob_fedu(df_unencoded)
fig_famsize_famrel_reason = dv.distribution_famsize_famrel_reason(df_unencoded)
fig_traveltime_studytime_schoolsup = dv.distribution_traveltime_studytime_schoolsup(df_unencoded)
fig_famsup_activities_paidclass = dv.distribution_famsup_activities_paidclass(df_unencoded)
fig_internet_nursery_higher = dv.distribution_internet_nursery_higher(df_unencoded)
fig_romantic_freetime_goout = dv.distribution_romantic_freetime_goout(df_unencoded)
fig_walc_dalc_health = dv.distribution_walc_dalc_health(df_unencoded)
fig_failures = dv.distribution_failures_column(df_unencoded)
fig_grade_age = dv.chart_grade_age(df_unencoded)
fig_grade_sex = dv.chart_grade_sex(df_unencoded)
fig_grade_school = dv.chart_grade_school(df_unencoded)
fig_grade_address = dv.chart_grade_address(df_unencoded)
fig_grade_pstatus = dv.chart_grade_pstatus(df_unencoded)
fig_grade_medu = dv.chart_grade_medu(df_unencoded)
fig_grade_mjob = dv.chart_grade_mjob(df_unencoded)
###
fig_grade_fedu = dv.chart_grade_fedu(df_unencoded)
fig_grade_fjob = dv.chart_grade_fedu(df_unencoded)
fig_grade_guardian = dv.chart_grade_guardian(df_unencoded)
fig_grade_famsize = dv.chart_grade_famsize(df_unencoded)
fig_grade_famrel = dv.chart_grade_famrel(df_unencoded)
fig_grade_reason = dv.chart_grade_reason(df_unencoded)
fig_grade_traveltime = dv.chart_grade_traveltime(df_unencoded)
fig_grade_studytime = dv.chart_grade_studytime(df_unencoded)
fig_grade_failures = dv.chart_grade_failures(df_unencoded)
fig_grade_schoolsup = dv.chart_grade_schoolsup(df_unencoded)
fig_grade_famsup = dv.chart_grade_famsup(df_unencoded)
fig_grade_activities = dv.chart_grade_activities(df_unencoded)
fig_grade_paidclass = dv.chart_grade_paidclass(df_unencoded)
fig_grade_internet = dv.chart_grade_internet(df_unencoded)
fig_grade_nursery = dv.chart_grade_nursery(df_unencoded)
fig_grade_higher = dv.chart_grade_higher(df_unencoded)
fig_grade_romantic = dv.chart_grade_romantic(df_unencoded)
fig_grade_freetime = dv.chart_grade_freetime(df_unencoded)
fig_grade_goout = dv.chart_grade_goout(df_unencoded)
fig_grade_walc = dv.chart_grade_walc(df_unencoded)
fig_grade_dalc = dv.chart_grade_dalc(df_unencoded)
fig_grade_health = dv.chart_grade_health(df_unencoded)
fig_grade_absences = dv.chart_grade_absences(df_unencoded)

# Making a slider for each significant value so user can manipulate and predict values
slider_1_label = df_feature_importances_g3.index[0]
slider_1_min = math.floor(df[slider_1_label].min())
slider_1_mean = round(df[slider_1_label].mean())
slider_1_max = round(df[slider_1_label].max())

slider_2_label = df_feature_importances_g3.index[1]
slider_2_min = math.floor(df[slider_2_label].min())
slider_2_mean = round(df[slider_2_label].mean())
slider_2_max = round(df[slider_2_label].max())

slider_3_label = df_feature_importances_g3.index[2]
slider_3_min = math.floor(df[slider_3_label].min())
slider_3_mean = round(df[slider_3_label].mean())
slider_3_max = round(df[slider_3_label].max())


##########################################################################################

# Making a slider for each significant value so user can manipulate and predict values
slider_4_label = df_feature_importances_g2.index[0]
slider_4_min = math.floor(df[slider_4_label].min())
slider_4_mean = round(df[slider_4_label].mean())
slider_4_max = round(df[slider_4_label].max())

slider_5_label = df_feature_importances_g2.index[1]
slider_5_min = math.floor(df[slider_5_label].min())
slider_5_mean = round(df[slider_5_label].mean())
slider_5_max = round(df[slider_5_label].max())

slider_6_label = df_feature_importances_g2.index[2]
slider_6_min = math.floor(df[slider_6_label].min())
slider_6_mean = round(df[slider_6_label].mean())
slider_6_max = round(df[slider_6_label].max())

######################################################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Div("DashBoard", className='header'),

    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    # ------------------------slider-1 container
                    html.Div([
                        html.H2(slider_1_label, className='heading'),
                        dcc.Slider(
                            id='slider_1', className='slider',
                            min=slider_1_min, max=slider_1_max, step=0.5, value=slider_1_mean,
                            marks={i: '{}'.format(i) for i in range(slider_1_min,slider_1_max+1)}
                        ),
                        html.Div(id='slider-1-output')
                    ]),
                    # ------------------------slider-2 container
                    html.Div([
                        html.H2(slider_2_label, className='heading'),
                        dcc.Slider(
                            id='slider_2', className='slider',
                            min=slider_2_min, max=slider_2_max, step=0.5, value=slider_2_mean,
                            marks={i: '{}'.format(i) for i in range(slider_2_min,slider_2_max+1)}
                        ),
                        html.Div(id='slider-2-output')
                    ]),
                    # ------------------------slider-3 container
                    html.Div([
                        html.H2(slider_3_label, className='heading'),
                        dcc.Slider(
                            id='slider_3',className='slider',
                            min=slider_3_min, max=slider_3_max, step=0.5, value=slider_3_mean,
                            marks={i: '{}'.format(i) for i in range(slider_3_min,slider_3_max+1)}
                        ),
                        html.Div(id='slider-3-output')
                    ]),
                    # ---------------------------------------Predicted score container
                    html.Div(html.P(id="prediction_result", className='score-div')),
                ], className='div-1'),
                # ---------------------------------------Row cols container
                html.Div([
                    html.Div(
                        [
                            dbc.Button(
                                "Details",
                                id="collapse-button", color="primary",
                                className="mb-3",
                            ),
                            dbc.Collapse(
                                dbc.Card(dbc.CardBody([html.P('Number Of Columns: '+str(number_of_cols), className='text-btn'),
                                                      html.P('Number Of Rows: '+str(number_of_rows), className='text-btn')]),
                                                      style={'margin': 8}),
                                id="collapse",
                            ),
                        ]
                    ),
                ], className='div-1')
            ], width=4),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=fig_features_importance_g3)
                ], className='div-2')
            ], width=8),
        ], no_gutters=True),
    ]),
    ##################################################################################
    # ------------------------------------------------------model detail container
    html.Div([
        html.Div(
            [
                dbc.Button(
                    "Details",
                    id="collapse-button-4", color="primary",
                    className="mb-3",
                ),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody(details_of_model_g3), style={'margin': 8}),
                    id="collapse-4",
                    style={'padding': 10}
                ),
            ]
        ),
    ], className='information-div'),
    ##################################################################################

    html.Div([
        html.Div(
            [
      
            ]
        ),
    ], className='space-div'),

    ##################################################################################
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    # ------------------------slider-1 container
                    html.Div([
                        html.H2(slider_4_label, className='heading'),
                        dcc.Slider(
                            id='slider_4', className='slider',
                            min=slider_4_min, max=slider_4_max, step=0.5, value=slider_4_mean,
                            marks={i: '{}'.format(i) for i in range(slider_4_min,slider_4_max+1)}
                        ),
                        html.Div(id='slider-4-output')
                    ]),
                    # ------------------------slider-2 container
                    html.Div([
                        html.H2(slider_5_label, className='heading'),
                        dcc.Slider(
                            id='slider_5', className='slider',
                            min=slider_5_min, max=slider_5_max, step=0.5, value=slider_5_mean,
                            marks={i: '{}'.format(i) for i in range(slider_5_min,slider_5_max+1)}
                        ),
                        html.Div(id='slider-5-output')
                    ]),
                    # ------------------------slider-3 container
                    html.Div([
                        html.H2(slider_6_label, className='heading'),
                        dcc.Slider(
                            id='slider_6',className='slider',
                            min=slider_6_min, max=slider_6_max, step=0.5, value=slider_6_mean,
                            marks={i: '{}'.format(i) for i in range(slider_6_min,slider_6_max+1)}
                        ),
                        html.Div(id='slider-6-output')
                    ]),
                    # ---------------------------------------Predicted score container
                    html.Div(html.P(id="prediction_result_G2", className='score-div')),
                ], className='div-1'),
                # ---------------------------------------Row cols container
                html.Div([
                    html.Div(
                        [
                            dbc.Button(
                                "Details",
                                id="collapse-button-3", color="primary",
                                className="mb-3",
                            ),
                            dbc.Collapse(
                                dbc.Card(dbc.CardBody([html.P('Number Of Columns: '+str(number_of_cols), className='text-btn'),
                                                      html.P('Number Of Rows: '+str(number_of_rows), className='text-btn')]),
                                                      style={'margin': 8}),
                                id="collapse-3",
                            ),
                        ]
                    ),
                ], className='div-1')
            ], width=4),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=fig_features_importance_g2)
                ], className='div-2')
            ], width=8),
        ], no_gutters=True),
    ]),

    # ------------------------------------------------------model detail container
    html.Div([
        html.Div(
            [
                dbc.Button(
                    "Details",
                    id="collapse-button-2", color="primary",
                    className="mb-3",
                ),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody(details_of_model_g2), style={'margin': 8}),
                    id="collapse-2",
                    style={'padding': 10}
                ),
            ]
        ),
    ], className='information-div'),

    ##################################################################################

    html.Div([
        html.Div(
            [
      
            ]
        ),
    ], className='space-div'),

    ##################################################################################


     # ---------------------------------------------------------subplots 1
    html.Div([
            dcc.Graph(figure=fig_correlation_heatmap,style={'margin': 10})
        ], className='bottom-plots'),

    # ---------------------------------------------------------subplots 2
    html.Div([
            dcc.Graph(figure=fig_boxplots_numeric_columns,style={'margin': 10})
        ], className='bottom-plots'),

    #-------------------------------------------------------------subplots 3
    html.Div([
            dcc.Graph(figure=fig_sex_mjob_school_columns, style={'margin': 10})
        ], className='bottom-plots'),

    #-------------------------------------------------------------subplots 4
    html.Div([
            dcc.Graph(figure=fig_address_pstatus_medu, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 5
    html.Div([
            dcc.Graph(figure=fig_guardian_fjob_fedu, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 6
    html.Div([
            dcc.Graph(figure=fig_famsize_famrel_reason, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 7
    html.Div([
            dcc.Graph(figure=fig_traveltime_studytime_schoolsup, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 8
    html.Div([
            dcc.Graph(figure=fig_famsup_activities_paidclass, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 9
    html.Div([
            dcc.Graph(figure=fig_internet_nursery_higher, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 10
    html.Div([
            dcc.Graph(figure=fig_romantic_freetime_goout, style={'margin': 10})
        ], className='bottom-plots'),

    #-------------------------------------------------------------subplots 11
    html.Div([
            dcc.Graph(figure=fig_walc_dalc_health, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 12
    html.Div([
            dcc.Graph(figure=fig_failures, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 13
    html.Div([
            dcc.Graph(figure=fig_grade_age, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 14
    html.Div([
            dcc.Graph(figure=fig_grade_sex, style={'margin': 10})
        ], className='bottom-plots'),

    #-------------------------------------------------------------subplots 15
    html.Div([
            dcc.Graph(figure=fig_grade_school, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 16
    html.Div([
            dcc.Graph(figure=fig_grade_address, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 17
    html.Div([
            dcc.Graph(figure=fig_grade_pstatus, style={'margin': 10})
        ], className='bottom-plots'),

    #-------------------------------------------------------------subplots 18
    html.Div([
            dcc.Graph(figure=fig_grade_medu, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 19
    html.Div([
            dcc.Graph(figure=fig_grade_mjob, style={'margin': 10})
        ], className='bottom-plots'),

     #-------------------------------------------------------------subplots 20
    html.Div([
            dcc.Graph(figure=fig_grade_fedu, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 21
    html.Div([
            dcc.Graph(figure=fig_grade_fjob, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 22
    html.Div([
            dcc.Graph(figure=fig_grade_guardian, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 23
    html.Div([
            dcc.Graph(figure=fig_grade_famsize, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 24
    html.Div([
            dcc.Graph(figure=fig_grade_famrel, style={'margin': 10})
        ], className='bottom-plots'),

     #-------------------------------------------------------------subplots 25
    html.Div([
            dcc.Graph(figure=fig_grade_reason, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 26
    html.Div([
            dcc.Graph(figure=fig_grade_traveltime, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 27
    html.Div([
            dcc.Graph(figure=fig_grade_studytime, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 28
    html.Div([
            dcc.Graph(figure=fig_grade_failures, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 29
    html.Div([
            dcc.Graph(figure=fig_grade_schoolsup, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 30
    html.Div([
            dcc.Graph(figure=fig_grade_famsup, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 31
    html.Div([
            dcc.Graph(figure=fig_grade_activities, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 32
    html.Div([
            dcc.Graph(figure=fig_grade_paidclass, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 33
    html.Div([
            dcc.Graph(figure=fig_grade_internet, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 34
    html.Div([
            dcc.Graph(figure=fig_grade_nursery, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 35
    html.Div([
            dcc.Graph(figure=fig_grade_higher, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 36
    html.Div([
            dcc.Graph(figure=fig_grade_romantic, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 37
    html.Div([
            dcc.Graph(figure=fig_grade_freetime, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 38
    html.Div([
            dcc.Graph(figure=fig_grade_goout, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 39
    html.Div([
            dcc.Graph(figure=fig_grade_walc, style={'margin': 10})
        ], className='bottom-plots'),


     #-------------------------------------------------------------subplots 40
    html.Div([
            dcc.Graph(figure=fig_grade_dalc, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 41
    html.Div([
            dcc.Graph(figure=fig_grade_health, style={'margin': 10})
        ], className='bottom-plots'),


    #-------------------------------------------------------------subplots 42
    html.Div([
            dcc.Graph(figure=fig_grade_absences, style={'margin': 10})
        ], className='bottom-plots')



], className='Main-background')

@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-2", "is_open"),
    [Input("collapse-button-2", "n_clicks")],
    [State("collapse-2", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-3", "is_open"),
    [Input("collapse-button-3", "n_clicks")],
    [State("collapse-3", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-4", "is_open"),
    [Input("collapse-button-4", "n_clicks")],
    [State("collapse-4", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open



# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("slider_1","value"), Input("slider_2","value"), Input("slider_3","value")])

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3):

    input_X = np.array([df["Mjob"].mean(),      
                       df["Fedu"].mean(),
                       df["school"].mean(),
                       df["sex"].mean(),
                       df["Medu"].mean(),
                       X3,
                       df["Pstatus"].mean(),
                       df["address"].mean(),
                       df["famsize"].mean(),
                       X2,
                       X1
                       ]).reshape(1,-1)        
    
    # Prediction is calculated based on the input_X array
    prediction = regressor_g3.predict(input_X)[0]
    
    # And retuned to the Output of the callback function
    return "Predicted Score G3: {}".format(round(prediction,1))


# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result_G2",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("slider_4","value"), Input("slider_5","value"), Input("slider_6","value")])

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3):
    input_X = np.array([X3,
                        X2,
                        df["studytime"].mean(),
                        X1
                       ]).reshape(1,-1)        
    
    # Prediction is calculated based on the input_X array
    prediction = regressor_g2.predict(input_X)[0]
    
    # And retuned to the Output of the callback function
    return "Predicted Score G2: {}".format(round(prediction,1))

if __name__ == '__main__':
    app.run_server(port=8050, host='0.0.0.0')
