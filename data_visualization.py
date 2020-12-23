import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Display boxplots of numeric values
def feature_importance_chart(df_feature_importances):
    if type(df_feature_importances).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig_features_importance = go.Figure()
    fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                             y=df_feature_importances["Importance"]
                                             ))
    fig_features_importance.update_layout(title_text='Features contribution in the model', 
                                          title_x=0.5,
                                          height=500,
                                          xaxis_title="Features",
                                          yaxis_title="Importance")
    return fig_features_importance


# Correlation heatmap of the whole dataframe
def correlation_heatmap(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    c=df.corr()
    fig = go.Figure(data=go.Heatmap(z=c.values,
                               x=['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout',
                                 'Dalc','Walc','health','absences','G1','G2','G3'],
                               y=['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout',
                                 'Dalc','Walc','health','absences','G1','G2','G3'],
                                colorscale='Inferno'
                               ))
    fig.update_layout(title_text='Correlation heatmap of the dataset', 
                                          title_x=0.5,
                                          height=500)
    return fig




def boxplots_numeric_columns(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df_numeric = df[['age','absences','G1','G2','G3']]
    fig_boxplot = go.Figure()
    fig_boxplot.add_trace(go.Box(y=df['G1'], name="G1"))
    fig_boxplot.add_trace(go.Box(y=df['G2'], name="G2"))
    fig_boxplot.add_trace(go.Box(y=df['G3'], name="G3"))
    fig_boxplot.add_trace(go.Box(y=df['absences'], name="absences"))
    fig_boxplot.add_trace(go.Box(y=df['age'], name="age"))
    fig_boxplot.update_layout(title_text='Boxplot of numeric columns', title_x=0.5, height=500,
                              xaxis_title="Column Variables",
                              yaxis_title="Distribution")
    return fig_boxplot


# Distribution of 'sex' column
def distribution_sex_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['sex'].value_counts().plot(kind='bar').set_title("Sex column distribution")
    return None


# Distribution of 'Mjob' column
def distribution_mjob_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['Mjob'].value_counts().plot(kind='bar').set_title("Mjob column distribution")
    return None


# Distribution of 'school' column
def distribution_school_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['school'].value_counts().plot(kind='bar').set_title("School column distribution")
    return None


# Distribution of 'sex', 'mjob', 'school'
def distribution_sex_mjob_school(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Sex", "School", "Mother job"))
    sex_x_axis = ['Female', 'Male']
    sex_y_axis = (df['sex'].value_counts()).tolist()
    school_x_axis = (df['school'].unique()).tolist()
    school_y_axis = (df['school'].value_counts()).tolist()
    mjob_x_axis = ['Other', 'Services', 'Teacher', 'At home', 'Health']
    mjob_y_axis = (df['Mjob'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=sex_x_axis, y=sex_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=school_x_axis, y=school_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=mjob_x_axis, y=mjob_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Sex, Mother job, School', title_x=0.5,height=500, showlegend=False)
    return fig


# Distribution of 'address' column
def distribution_address_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['address'].value_counts().plot(kind='bar').set_title("Address column distribution")
    return None


# Distribution of 'Pstatus' column
def distribution_pstatus_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['Pstatus'].value_counts().plot(kind='bar').set_title("Pstatus column distribution")
    return None


# Distribution of 'Medu' column
def distribution_medu_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['Medu'].value_counts().plot(kind='bar').set_title("Medu column distribution")
    return None


# Distribution of 'address', 'pstatus', 'medu'
def distribution_address_pstatus_medu(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Address", "Parents status", "Mother education"))
    address_x_axis = ["Urban", "Rural"]
    address_y_axis = (df['address'].value_counts()).tolist()
    Pstatus_x_axis = ["Living Together", "Apart"]
    Pstatus_y_axis = (df['Pstatus'].value_counts()).tolist()
    Medu_x_axis = ['Higher Education', 'Secondary Education', '5-9th Grade', '4th Grade', 'None']
    Medu_y_axis = (df['Medu'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=address_x_axis, y=address_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=Pstatus_x_axis, y=Pstatus_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=Medu_x_axis, y=Medu_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Address, Parents status, Mothers education', title_x=0.5, height=500, showlegend=False)
    return fig


# Distribution of 'Fjob' column 
def distribution_fjob_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['Fjob'].value_counts().plot(kind='bar').set_title("Fjob column distribution")
    return None


# Distribution of 'Fedu' column
def distribution_fedu_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['Fedu'].value_counts().plot(kind='bar').set_title("Fedu column distribution")
    return None



# Distribution of 'guardian' column
def distribution_guardian_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['guardian'].value_counts().plot(kind='bar').set_title("guardian column distribution")
    return None


# Distribution of 'Guardian', 'Fjob', 'Fedu'
def distribution_guardian_fjob_fedu(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Guardian", "Father job", "Father education"))
    guardian_x_axis = ['Mother', 'Father', 'Other']
    guardian_y_axis = (df['guardian'].value_counts()).tolist()
    fjob_x_axis = ['Other', 'Services', 'Teacher', 'At Home', 'Health']
    fjob_y_axis = (df['Fjob'].value_counts()).tolist()
    fedu_x_axis = ['5-9th Grade', 'Secondary Education', 'Higher Education', 'Primary Education', 'None']
    fedu_y_axis = (df['Fedu'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=guardian_x_axis, y=guardian_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=fjob_x_axis, y=fjob_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=fedu_x_axis, y=fedu_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Guardian, Father job, Father education', title_x=0.5, height=500, showlegend=False)
    return fig


# Distribution of 'famsize' column
def distribution_famsize_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['famsize'].value_counts().plot(kind='bar').set_title("Famsize column distribution")
    return None


# Distribution of 'famrel' column
def distribution_famrel_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['famrel'].value_counts().plot(kind='bar').set_title("Famrel column distribution")
    return None


# Distribution of 'reason' column
def distribution_reason_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['reason'].value_counts().plot(kind='bar').set_title("Reason column distribution")
    return None


# Distribution of 'Famsize', 'Famrel', 'Reason'
def distribution_famsize_famrel_reason(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Family Size", "Family Relations", "Reason to choose this school"))
    famsize_x_axis = ['Greater than 3', 'Less than or equal to 3']
    famsize_y_axis = (df['famsize'].value_counts()).tolist()
    famrel_x_axis = [4,5,3,2,1]
    famrel_y_axis = (df['famrel'].value_counts()).tolist()
    reason_x_axis = ['Course', 'Home', 'Reputation', 'Other']
    reason_y_axis = (df['reason'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=famsize_x_axis, y=famsize_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=famrel_x_axis, y=famrel_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=reason_x_axis, y=reason_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Family Size, Family Relations, Reason', title_x=0.5, height=500, showlegend=False)
    return fig


# Distribution of 'traveltime' column
def distribution_traveltime_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['traveltime'].value_counts().plot(kind='bar').set_title("Traveltime column distribution")
    return None


# Distribution of 'studytime' column
def distribution_studytime_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['studytime'].value_counts().plot(kind='bar').set_title("Studytime column distribution")
    return None


# Distribution of 'schoolsup' column
def distribution_schoolsup_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['schoolsup'].value_counts().plot(kind='bar').set_title("Extra education support column distribution")
    return None



# Distribution of 'traveltime', 'studytime', 'schoolsup'
def distribution_traveltime_studytime_schoolsup(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Travel Time", "Study Time", "Extra educational school Support"))
    traveltime_x_axis = ["15mins", "15-30mins", "30mins", "1 hour", ">1 hour"]
    traveltime_y_axis = (df['traveltime'].value_counts()).tolist()
    studytime_x_axis = ['2-5hrs','<2hrs', '5-10hrs', '>10hrs']
    studytime_y_axis = (df['studytime'].value_counts()).tolist()
    schoolsup_x_axis = ['Yes', 'No']
    schoolsup_y_axis = (df['schoolsup'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=traveltime_x_axis, y=traveltime_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=studytime_x_axis, y=studytime_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=schoolsup_x_axis, y=schoolsup_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Travel time, Study time, Extra educational school support', title_x=0.5, height=500, showlegend=False)
    return fig


# Distribution of 'famsup' column
def distribution_famsup_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['famsup'].value_counts().plot(kind='bar').set_title("Family support column distribution")
    return None


# Distribution of 'activities' column
def distribution_activities_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['activities'].value_counts().plot(kind='bar').set_title("Activities column distribution")
    return None


# Distribution of 'paidclass' column
def distribution_paidclass_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['paidclass'].value_counts().plot(kind='bar').set_title("Extra paid classes column distribution")
    return None


# Distribution of 'famsup', 'activities', 'paidclass'
def distribution_famsup_activities_paidclass(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Family Support", "Activities", "Paid Class"))
    famsup_x_axis = ["Yes","No"]
    famsup_y_axis = (df['famsup'].value_counts()).tolist()
    activities_x_axis = ['Yes','No']
    activities_y_axis = (df['studytime'].value_counts()).tolist()
    paidclass_x_axis = ['No', 'Yes']
    paidclass_y_axis = (df['paid'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=famsup_x_axis, y=famsup_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=activities_x_axis, y=activities_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=paidclass_x_axis, y=paidclass_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Family Support, Activities, Paid class', title_x=0.5, height=500, showlegend=False)
    return fig



# Distribution of 'internet' column
def distribution_internet_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['internet'].value_counts().plot(kind='bar').set_title("Internet column distribution")
    return None


# Distribution of 'nursery' column
def distribution_nursery_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['nursery'].value_counts().plot(kind='bar').set_title("Nursery column distribution")
    return None


# Distribution of 'higher' column
def distribution_higher_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['higher'].value_counts().plot(kind='bar').set_title("Higher education column distribution")
    return None


# Distribution of 'internet', 'nursery', 'higher'
def distribution_internet_nursery_higher(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Internet at home", "Attended nursery", "Interested in higher education"))
    internet_x_axis = ["Yes","No"]
    internet_y_axis = (df['internet'].value_counts()).tolist()
    nursery_x_axis = ['Yes','No']
    nursery_y_axis = (df['nursery'].value_counts()).tolist()
    higher_x_axis = ['Yes', 'No']
    higher_y_axis = (df['higher'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=internet_x_axis, y=internet_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=nursery_x_axis, y=nursery_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=higher_x_axis, y=higher_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Internet, Nursery, Higher', title_x=0.5, height=500, showlegend=False)
    return fig



# Distribution of 'romantic' column
def distribution_romantic_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['romantic'].value_counts().plot(kind='bar').set_title("Romantic relation column distribution")
    return None


# Distribution of 'freetime' column
def distribution_freetime_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['freetime'].value_counts().plot(kind='bar').set_title("Freetime column distribution")
    return None


# Distribution of 'goout' column
def distribution_goout_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['goout'].value_counts().plot(kind='bar').set_title("Going out column distribution")
    return None


# Distribution of 'romantic', 'freetime', 'goout'
def distribution_romantic_freetime_goout(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("In a romantic relationship", "Freetime after school", "Going out with friends"))
    romantic_x_axis = ["No", "Yes"]
    romantic_y_axis = (df['romantic'].value_counts()).tolist()
    freetime_x_axis = ['Medium', 'Low', 'High', 'Very High', 'Very Low' ]
    freetime_y_axis = (df['freetime'].value_counts()).tolist()
    higher_x_axis = ['Yes', 'No']
    higher_y_axis = (df['higher'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=romantic_x_axis, y=romantic_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=freetime_x_axis, y=freetime_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=higher_x_axis, y=higher_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Romantic, Freetime, GoOut', title_x=0.5, height=500, showlegend=False)
    return fig



# Distribution of 'Walc' column
def distribution_walc_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['Walc'].value_counts().plot(kind='bar').set_title("Weekend alcohol column distribution")
    return None


# Distribution of 'Dalc' column
def distribution_dalc_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['Dalc'].value_counts().plot(kind='bar').set_title("Weekday alcohol column distribution")
    return None


# Distribution of 'health' column
def distribution_health_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['health'].value_counts().plot(kind='bar').set_title("Health column distribution")
    return None


# Distribution of 'walc', 'dalc', 'health'
def distribution_walc_dalc_health(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Weekend alcohol consumption", "Workday alcohol consumption", "Current health status"))
    walc_x_axis = ["Very Low","Low","Medium","High","Very High"]
    walc_y_axis = (df['Walc'].value_counts()).tolist()
    dalc_x_axis = ["Very Low","Low","Medium","High","Very High"]
    dalc_y_axis = (df['Dalc'].value_counts()).tolist()
    health_x_axis = ['Very Good', 'Medium', 'Good', 'Very Bad', 'Bad']
    health_y_axis = (df['health'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=walc_x_axis, y=walc_y_axis),row=1,col=1)
    fig.add_trace(go.Bar(x=dalc_x_axis, y=dalc_y_axis),row=1,col=2)
    fig.add_trace(go.Bar(x=health_x_axis, y=health_y_axis),row=1,col=3)
    fig.update_layout(title_text='Distribution plots of Weekend alcohol consumption, Workday alcohol consumption, Current health status', title_x=0.5, height=500, showlegend=False)
    return fig

    
# Distribution of 'age' column
def distribution_age_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df['age'].value_counts().plot(kind='bar').set_title("Age column distribution")
    return None


# Distribution of 'absences' column
def distribution_absences_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    sns.distplot(df['absences']).set_title("Absences column distribution")
    return None


# Distribution of 'failures' column
def distribution_failures_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    x_labels = [0,1,2,3]
    y_labels = (df['failures'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_labels, y=y_labels))
    fig.update_layout(title_text='Distribution plots of Failures', title_x=0.5, height=500, width=600)
    return fig


# Distribution of 'G1' column
def distribution_G1_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    sns.distplot(df['G1']).set_title("G1 column distribution")
    return None


# Distribution of 'G2' column
def distribution_G2_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    sns.distplot(df['G2']).set_title("G2 column distribution")
    return None


# Distribution of 'G3' column
def distribution_G3_column(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    sns.distplot(df['G3']).set_title("G3 column distribution")
    return None


# Heatmap of the dataset
def heatmap(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    c= df.corr()
    plt.figure(figsize=(16,8))
    sns.heatmap(c,annot=True).set_title("Heatmap for correlation overview")
    return None



# Doing scatter plots for the grade against other variables

# Plot for 'age' column against grades
def chart_grade_age(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Age", "G2 vs Age", "G3 vs Age"))
    fig.add_trace(go.Scatter(x=df['age'], y=df['G1'],mode='markers'),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['age'], y=df['G2'],mode='markers'),row=1,col=2)
    fig.add_trace(go.Scatter(x=df['age'], y=df['G3'],mode='markers'),row=1,col=3)
    fig.update_layout(title_text='Charts of Grades against Age', title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_xaxes(title_text="Age", row=1, col=2)
    fig.update_xaxes(title_text="Age", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'sex' column against grades
def chart_grade_sex(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Sex", "G2 vs Sex", "G3 vs Sex"))
    df["sex"].replace({"M": "Male", "F": "Female"}, inplace=True)
    fig.add_trace(go.Box(x=df['sex'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['sex'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['sex'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text='Charts of Grades against Sex', title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Sex", row=1, col=1)
    fig.update_xaxes(title_text="Sex", row=1, col=2)
    fig.update_xaxes(title_text="Sex", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'school' column against grades
def chart_grade_school(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs School", "G2 vs School", "G3 vs School"))
    fig.add_trace(go.Box(x=df['school'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['school'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['school'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text='Charts of Grades against School', title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="School", row=1, col=1)
    fig.update_xaxes(title_text="School", row=1, col=2)
    fig.update_xaxes(title_text="School", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'address' column against grades
def chart_grade_address(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Address", "G2 vs Address", "G3 vs Address"))
    df["address"].replace({"U": "Urban", "R": "Rural"}, inplace=True)
    fig.add_trace(go.Box(x=df['address'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['address'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['address'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text='Charts of Grades against Address', title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Address", row=1, col=1)
    fig.update_xaxes(title_text="Address", row=1, col=2)
    fig.update_xaxes(title_text="Address", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'Pstatus' column against grades
def chart_grade_pstatus(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Parent's status", "G2 vs Parent's status", "G3 vs Parent's status"))
    df["Pstatus"].replace({"A": "Apart", "T": "Together"}, inplace=True)
    fig.add_trace(go.Box(x=df['Pstatus'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['Pstatus'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['Pstatus'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Parent's status", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Parent's Status", row=1, col=1)
    fig.update_xaxes(title_text="Parent's Status", row=1, col=2)
    fig.update_xaxes(title_text="Parent's Status", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'Medu' column against grades
def chart_grade_medu(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Mother's education", "G2 vs Mother's education", "G3 vs Mother's education"))
    df["Medu"].replace({0: "None", 1: "4th Grade", 2: "5-9th Grade", 3: "Secondary", 4: "Higher"}, inplace=True)
    fig.add_trace(go.Box(x=df['Medu'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['Medu'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['Medu'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Mother's education", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Mother's Education", row=1, col=1)
    fig.update_xaxes(title_text="Mother's Education", row=1, col=2)
    fig.update_xaxes(title_text="Mother's Education", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig



# Plot for 'Mjob' column against grades
def chart_grade_mjob(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Mother's job", "G2 vs Mother's job", "G3 vs Mother's job"))
    df["Mjob"].replace({"at_home": "At Home", "health": "Health", "other": "Other", "services": "Services", "teacher": "Teacher"}, inplace=True)
    fig.add_trace(go.Box(x=df['Mjob'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['Mjob'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['Mjob'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Mother's job", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Mother's Job", row=1, col=1)
    fig.update_xaxes(title_text="Mother's Job", row=1, col=2)
    fig.update_xaxes(title_text="Mother's Job", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'Fedu' column against grades
def chart_grade_fedu(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Father's education", "G2 vs Father's education", "G3 vs Father's education"))
    df["Fedu"].replace({0: "None", 1: "4th Grade", 2: "5-9th Grade", 3: "Secondary", 4: "Higher"}, inplace=True)
    fig.add_trace(go.Box(x=df['Fedu'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['Fedu'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['Fedu'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Father's education", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Father's Education", row=1, col=1)
    fig.update_xaxes(title_text="Father's Education", row=1, col=2)
    fig.update_xaxes(title_text="Father's Education", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'Fjob' column against grades
def chart_grade_fjob(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Father's job", "G2 vs Father's job", "G3 vs Father's job"))
    df["Fjob"].replace({"at_home": "At Home", "health": "Health", "other": "Other", "services": "Services", "teacher": "Teacher"}, inplace=True)
    fig.add_trace(go.Box(x=df['Fjob'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['Fjob'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['Fjob'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Father's job", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Father's Job", row=1, col=1)
    fig.update_xaxes(title_text="Father's Job", row=1, col=2)
    fig.update_xaxes(title_text="Father's Job", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'guardian' column against grades
def chart_grade_guardian(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Gaurdian", "G2 vs Guardian", "G3 vs Guardian"))
    df["guardian"].replace({"mother": "Mother", "father": "Father", "other": "Other"}, inplace=True)
    fig.add_trace(go.Box(x=df['guardian'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['guardian'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['guardian'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Guardian", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Guardian", row=1, col=1)
    fig.update_xaxes(title_text="Guardian", row=1, col=2)
    fig.update_xaxes(title_text="Guardian", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'famsize' column against grades
def chart_grade_famsize(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Family size", "G2 vs Family size", "G3 vs Family size"))
    df["famsize"].replace({"GT3": ">3", "LE3": "<=3"}, inplace=True)
    fig.add_trace(go.Box(x=df['famsize'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['famsize'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['famsize'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Family Size", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Family Size", row=1, col=1)
    fig.update_xaxes(title_text="Family Size", row=1, col=2)
    fig.update_xaxes(title_text="Family Size", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'famsize' column against grades
def chart_grade_famrel(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Family relations", "G2 vs Family relations", "G3 vs Family relations"))
    df["famrel"].replace({1: "Very Bad", 2: "Bad", 3: "Medium", 4: "Good", 5: "Excellent"}, inplace=True)
    fig.add_trace(go.Box(x=df['famrel'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['famrel'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['famrel'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Family Relations", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Family Relations", row=1, col=1)
    fig.update_xaxes(title_text="Family Relations", row=1, col=2)
    fig.update_xaxes(title_text="Family Relations", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'reason' column against grades
def chart_grade_reason(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Reason", "G2 vs Reason", "G3 vs Reason"))
    df["reason"].replace({"close to home": "Close to home", "school reputation": "School Reputation", "course preference": "Course Preference", "other": "Other"}, inplace=True)
    fig.add_trace(go.Box(x=df['reason'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['reason'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['reason'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Reason to chose this school", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Reason", row=1, col=1)
    fig.update_xaxes(title_text="Reason", row=1, col=2)
    fig.update_xaxes(title_text="Reason", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig



# Plot for 'traveltime' column against grades
def chart_grade_traveltime(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Travel time", "G2 vs Travel time", "G3 vs Travel time"))
    df["traveltime"].replace({1: "<15mins", 2: "15-30mins", 3: "30mins-1hr", 4: ">1hr"}, inplace=True)
    fig.add_trace(go.Box(x=df['traveltime'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['traveltime'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['traveltime'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Travel time", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Travel time", row=1, col=1)
    fig.update_xaxes(title_text="Travel time", row=1, col=2)
    fig.update_xaxes(title_text="Travel time", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'studytime' column against grades
def chart_grade_studytime(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Study time", "G2 vs Study time", "G3 vs Study time"))
    df["studytime"].replace({1: "<2hrs", 2: "2-5hrs", 3: "5-10hrs", 4: ">10hrs"}, inplace=True)
    fig.add_trace(go.Box(x=df['studytime'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['studytime'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['studytime'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Study time", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Study time", row=1, col=1)
    fig.update_xaxes(title_text="Study time", row=1, col=2)
    fig.update_xaxes(title_text="Study time", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'failures' column against grades
def chart_grade_failures(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Failures", "G2 vs Failures", "G3 vs Failures"))
    fig.add_trace(go.Box(x=df['failures'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['failures'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['failures'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Failures", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Failures", row=1, col=1)
    fig.update_xaxes(title_text="Failures", row=1, col=2)
    fig.update_xaxes(title_text="Failures", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'schoolsup' column against grades
def chart_grade_schoolsup(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs School support", "G2 vs School support", "G3 vs School support"))
    df["schoolsup"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['schoolsup'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['schoolsup'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['schoolsup'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Extra educational school support ", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="School support", row=1, col=1)
    fig.update_xaxes(title_text="School support", row=1, col=2)
    fig.update_xaxes(title_text="School support", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'famsup' column against grades
def chart_grade_famsup(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Family support", "G2 vs Family support", "G3 vs Family support"))
    df["famsup"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['famsup'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['famsup'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['famsup'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Family educational support", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Family support", row=1, col=1)
    fig.update_xaxes(title_text="Family support", row=1, col=2)
    fig.update_xaxes(title_text="Family support", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'activities' column against grades
def chart_grade_activities(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Activities", "G2 vs Activities", "G3 vs Activities"))
    df["activities"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['activities'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['activities'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['activities'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Extra-curricular activities", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Extra activities", row=1, col=1)
    fig.update_xaxes(title_text="Extra activities", row=1, col=2)
    fig.update_xaxes(title_text="Extra activities", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'paidclass' column against grades
def chart_grade_paidclass(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Extra paid class", "G2 vs Extra paid class", "G3 vs Extra paid class"))
    df["paid"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['paid'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['paid'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['paid'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Extra paid classes", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Extra paid class", row=1, col=1)
    fig.update_xaxes(title_text="Extra paid class", row=1, col=2)
    fig.update_xaxes(title_text="Extra paid class", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig



# Plot for 'internet' column against grades
def chart_grade_internet(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Internet access", "G2 vs Internet access", "G3 vs Internet access"))
    df["internet"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['internet'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['internet'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['internet'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Internet access at home", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Internet access", row=1, col=1)
    fig.update_xaxes(title_text="Internet access", row=1, col=2)
    fig.update_xaxes(title_text="Internet access", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'nursery' column against grades
def chart_grade_nursery(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Nursery education", "G2 vs Nursery education", "G3 vs Nursery education"))
    df["nursery"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['nursery'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['nursery'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['nursery'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Attended nursery school", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Nursery education", row=1, col=1)
    fig.update_xaxes(title_text="Nursery education", row=1, col=2)
    fig.update_xaxes(title_text="Nursery education", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'higher' column against grades
def chart_grade_higher(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Higher education desire", "G2 vs Higher education desire", "G3 vs Higher education desire"))
    df["higher"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['higher'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['higher'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['higher'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against higher education desire", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Higher education desire", row=1, col=1)
    fig.update_xaxes(title_text="Higher education desire", row=1, col=2)
    fig.update_xaxes(title_text="Higher education desire", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'romantic' column against grades
def chart_grade_romantic(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Romantic relationship", "G2 vs Romantic relationship", "G3 vs Romantic relationship"))
    df["romantic"].replace({"yes": "Yes", "no": "No"}, inplace=True)
    fig.add_trace(go.Box(x=df['romantic'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['romantic'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['romantic'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Romantic relationship", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Romantic relationship", row=1, col=1)
    fig.update_xaxes(title_text="Romantic relationship", row=1, col=2)
    fig.update_xaxes(title_text="Romantic relationship", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'freetime' column against grades
def chart_grade_freetime(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Free time", "G2 vs Free time", "G3 vs Free time"))
    df["freetime"].replace({1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}, inplace=True)
    fig.add_trace(go.Box(x=df['freetime'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['freetime'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['freetime'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Free time after school", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Free time", row=1, col=1)
    fig.update_xaxes(title_text="Free time", row=1, col=2)
    fig.update_xaxes(title_text="Free time", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'goout' column against grades
def chart_grade_goout(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Going out", "G2 vs Going out", "G3 vs Going out"))
    df["goout"].replace({1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}, inplace=True)
    fig.add_trace(go.Box(x=df['goout'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['goout'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['goout'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Going out with friends", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Going out", row=1, col=1)
    fig.update_xaxes(title_text="Going out", row=1, col=2)
    fig.update_xaxes(title_text="Going out", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'walc' column against grades
def chart_grade_walc(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Weekend alcohol consumption", "G2 vs Weekend alcohol consumption", "G3 vs Weekend alcohol consumption"))
    df["Walc"].replace({1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}, inplace=True)
    fig.add_trace(go.Box(x=df['Walc'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['Walc'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['Walc'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Weekend alcohol consumption", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Weekend alcohol consumption", row=1, col=1)
    fig.update_xaxes(title_text="Weekend alcohol consumption", row=1, col=2)
    fig.update_xaxes(title_text="Weekend alcohol consumption", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'dalc' column against grades
def chart_grade_dalc(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Workday alcohol consumption", "G2 vs Workday alcohol consumption", "G3 vs Workday alcohol consumption"))
    df["Dalc"].replace({1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}, inplace=True)
    fig.add_trace(go.Box(x=df['Dalc'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['Dalc'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['Dalc'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Workday alcohol consumption", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Workday alcohol consumption", row=1, col=1)
    fig.update_xaxes(title_text="Workday alcohol consumption", row=1, col=2)
    fig.update_xaxes(title_text="Workday alcohol consumption", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig


# Plot for 'health' column against grades
def chart_grade_health(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Health status", "G2 vs Health status", "G3 vs Health status"))
    df["health"].replace({1: "Very Bad", 2: "Bad", 3: "Medium", 4: "Good", 5: "Very Good"}, inplace=True)
    fig.add_trace(go.Box(x=df['health'], y=df['G1']),row=1,col=1)
    fig.add_trace(go.Box(x=df['health'], y=df['G2']),row=1,col=2)
    fig.add_trace(go.Box(x=df['health'], y=df['G3']),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against Current health status", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Health status", row=1, col=1)
    fig.update_xaxes(title_text="Health status", row=1, col=2)
    fig.update_xaxes(title_text="Health status", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig



# Plot for 'absences' column against grades
def chart_grade_absences(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("G1 vs Absences", "G2 vs Absences", "G3 vs Absences"))
    fig.add_trace(go.Scatter(x=df['absences'], y=df['G1'], mode="markers"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['absences'], y=df['G2'], mode="markers"),row=1,col=2)
    fig.add_trace(go.Scatter(x=df['absences'], y=df['G3'], mode="markers"),row=1,col=3)
    fig.update_layout(title_text="Charts of Grades against School absences", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Absences", row=1, col=1)
    fig.update_xaxes(title_text="Absences", row=1, col=2)
    fig.update_xaxes(title_text="Absences", row=1, col=3)
    fig.update_yaxes(title_text="G1", row=1, col=1)
    fig.update_yaxes(title_text="G2", row=1, col=2)
    fig.update_yaxes(title_text="G3", row=1, col=3)
    return fig