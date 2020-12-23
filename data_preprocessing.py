import pandas as pd
from pathlib import Path
from scipy.stats.stats import pearsonr
import itertools

# To check whether file exists or not!
def check_file_existence(filepath):
    #"/Users/siddiqui/Documents/university/thesis/dataset/student/student-mat.csv"
    my_file = Path(filepath)
    if not(my_file.is_file()):
        raise Exception("File doesn't exist!")


# To read the file!
def read_file(filepath):
    df = pd.read_csv(filepath,sep=';')
    return df


# To check missing data!
def check_missing_data(df):
    # Checking for missing data
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    missing_data = False
    NaN_rows = df.isnull()
    NaN_rows = NaN_rows.any(axis=1)
    df_with_NaN = df[NaN_rows]
    df_with_NaN.shape[0]
    if(df_with_NaN.shape[0] != 0):
        raise Exception("Sorry, some data is missing!")
    return missing_data

# To get number of rows
def number_of_rows(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    return df.shape[0]


# To get number of columns
def number_of_cols(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    return df.shape[1]


# To display column names!
def display_column_names(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    return df.columns.tolist()



# Checking for rows with outliers in absences column
def row_with_outliers(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df_numeric = df[['absences','G1','G2','G3']]
    # IQR for absences column
    Q1_absences = df_numeric['absences'].quantile(0.25)
    Q3_absences = df_numeric['absences'].quantile(0.75)
    IQR_absences = Q3_absences - Q1_absences
    # IQR for G1 column
    Q1_G1 = df_numeric['G1'].quantile(0.25)
    Q3_G1 = df_numeric['G1'].quantile(0.75)
    IQR_G1 = Q3_G1 - Q1_G1
    # IQR for G2 column
    Q1_G2 = df_numeric['G2'].quantile(0.25)
    Q3_G2 = df_numeric['G2'].quantile(0.75)
    IQR_G2 = Q3_G2 - Q1_G2
    # IQR for G3 column
    Q1_G3 = df_numeric['G3'].quantile(0.25)
    Q3_G3 = df_numeric['G3'].quantile(0.75)
    IQR_G3 = Q3_G3 - Q1_G3
    Absences_Outliers1 = sum(df_numeric['absences'] < (Q1_absences - 1.5 * IQR_absences))
    Absences_Outliers2 = sum(df_numeric['absences'] > (Q3_absences + 1.5 * IQR_absences))
    # Checking for rows with outliers in G1 column
    G1_Outliers1 = sum(df_numeric['G1'] < (Q1_G1 - 1.5 * IQR_G1))
    G1_Outliers2 = sum(df_numeric['G1'] > (Q3_G1 + 1.5 * IQR_G1))
    # Checking for rows with outliers in G2 column
    G2_Outliers1 = sum(df_numeric['G2'] < (Q1_G2 - 1.5 * IQR_G2))
    G2_Outliers2 = sum(df_numeric['G2'] > (Q3_G2 + 1.5 * IQR_G2))
    # Checking for rows with outliers in G3 column
    G3_Outliers1 = sum(df_numeric['G3'] < (Q1_G3 - 1.5 * IQR_G3))
    G3_Outliers2 = sum(df_numeric['G3'] > (Q3_G3 + 1.5 * IQR_G3))
    outliers_dict = {
        "Absences_Outliers1": Absences_Outliers1,
        "Absences_Outliers2": Absences_Outliers2,
        "G1_Outliers1": G1_Outliers1,
        "G1_Outliers2": G1_Outliers2,
        "G2_Outliers1": G2_Outliers1,
        "G2_Outliers2": G2_Outliers2,
        "G3_Outliers1": G3_Outliers1,
        "G3_Outliers2": G3_Outliers2,
    }
    return outliers_dict



# Plausibility check
def plausibility_check(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    # Checking for any values which are negative in the numeric columns and might affect the output
    Sex_invalid_values = sum((df['sex'] != "F") & (df['sex'] != "M"))
    Age_invalid_values = sum(df['age'] < 15)
    Age_invalid_values2 = sum(df['age'] > 22)
    School_invalid_values = sum((df['school'] != "GP") & (df['school'] != "MS"))
    Address_invalid_values = sum((df['address'] != "U") & (df['address'] != "R"))
    Pstatus_invalid_values = sum((df['Pstatus'] != "A") & (df["Pstatus"] != "T"))
    Medu_invalid_values = sum((df['Medu'] < 0) & (df['Medu'] > 4))
    Mjob_invalid_values = sum((df['Mjob'] != "other") & (df['Mjob'] != "services") & (df['Mjob'] != "at_home") & (df['Mjob'] != "teacher") & (df['Mjob'] != "health"))
    Fedu_invalid_values =sum((df['Fedu'] < 0) & (df['Fedu'] > 4))
    Fjob_invalid_values = sum((df['Fjob'] != "other") & (df['Fjob'] != "services") & (df['Fjob'] != "at_home") & (df['Fjob'] != "teacher") & (df['Fjob'] != "health"))
    Guardian_invalid_values = sum((df['guardian'] != "mother") & (df['guardian'] != "father") & (df['guardian'] != "other"))
    Famsize_invalid_values = sum((df['famsize'] != "GT3") & (df['famsize'] != "LE3"))
    Famrel_invalid_values = sum((df['famrel'] < 1) & (df['famrel'] > 5))
    Reason_invalid_values = sum((df['reason'] != "course") & (df['reason'] != "home") & (df['reason'] != "reputation") & (df['reason'] != "other"))
    Traveltime_invalid_values = sum((df['traveltime'] < 1) & (df['traveltime'] > 4))
    Studytime_invalid_values = sum((df['studytime'] < 1) & (df['studytime'] > 4))
    Failures_invalid_values = sum((df['failures'] < 0) & (df['failures'] > 3))
    Schoolsup_invalid_values = sum((df['schoolsup'] != "yes") & (df['schoolsup'] != "no"))
    Famsup_invalid_values = sum((df['famsup'] != "yes") & (df['famsup'] != "no"))
    Activites_invalid_values = sum((df['activities'] != "yes") & (df['activities'] != "no"))
    Paidclass_invalid_values = sum((df['paid'] != "yes") & (df['paid'] != "no"))
    Internet_invalid_values = sum((df['internet'] != "yes") & (df['internet'] != "no"))
    Nursery_invalid_values = sum((df['nursery'] != "yes") & (df['nursery'] != "no"))
    Higher_invalid_values = sum((df['higher'] != "yes") & (df['higher'] != "no"))
    Romantic_invalid_values = sum((df['romantic'] != "yes") & (df['romantic'] != "no"))
    Freetime_invalid_values = sum((df['freetime'] < 1) & (df['freetime'] > 5))
    Goout_invalid_values = sum((df['goout'] < 1) & (df['goout'] > 5))
    Walc_invalid_values = sum((df['Walc'] < 1) & (df['Walc'] > 5))
    Dalc_invalid_values = sum((df['Dalc'] < 1) & (df['Dalc'] > 5))
    Health_invalid_values = sum((df['health'] < 1) & (df['health'] > 5))
    Absences_negative_values = sum(df['absences'] < 0)
    G1_negative_values = sum(df['G1'] < 0)
    G2_negative_values = sum(df['G2'] < 0)
    G3_negative_values = sum(df['G3'] < 0)
    
    plausibility_dict = {
    "Invalid values in sex": Sex_invalid_values,
    "Age lower invalid values": Age_invalid_values,
    "Age higher invalid values": Age_invalid_values2,
    "School invalid values": School_invalid_values,
    "Address invalid values": Address_invalid_values,
    "Pstatus invalid values": Pstatus_invalid_values,
    "Medu invalid values": Medu_invalid_values,
    "Mjob invalid values": Mjob_invalid_values,
    "Fedu invalid values": Fedu_invalid_values,
    "Fjob invalid values": Fjob_invalid_values,
    "Guardian invalid values": Guardian_invalid_values,
    "Famsize invalid values": Famsize_invalid_values,
    "Famrel invalid values": Famrel_invalid_values,
    "Reason invalid values": Reason_invalid_values,
    "Travel time invalid values": Traveltime_invalid_values,
    "Study time invalid values": Studytime_invalid_values,
    "Failures invalid values": Failures_invalid_values,
    "Schoolsup invalid values": Schoolsup_invalid_values,
    "Famsup invalid values": Famsup_invalid_values,
    "Activities invalid values": Activites_invalid_values,
    "Paidclass invalid values": Paidclass_invalid_values,
    "Internet invalid values": Internet_invalid_values,
    "Nursery invalid values": Nursery_invalid_values,
    "Higher invalid values": Higher_invalid_values,
    "Romantic invalid values": Romantic_invalid_values,
    "Freetime invalid values": Freetime_invalid_values,
    "Goout invalid values": Goout_invalid_values,
    "Walc invalid values": Walc_invalid_values,
    "Dalc invalid values": Dalc_invalid_values,
    "Health invalid values": Health_invalid_values,
    "Negative values in absence": Absences_negative_values,
    "Negative values in G1": G1_negative_values,
    "Negative values in G2": G2_negative_values,
    "Negative values in G3": G3_negative_values,
    }
    any_wrong_value = False
    for key in plausibility_dict:
        if plausibility_dict[key]>0:
            any_wrong_value = True
            raise Exception("Sorry, the values are invalid in dataframe!")
            break
    
    return any_wrong_value



# Calculating correlations between all numeric columns
def correlation_check(df):    
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    df_integer_columns = df.select_dtypes(include=['int64'])
    correlations = {}
    columns = df_integer_columns.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + '__' + col_b] = pearsonr(df_integer_columns.loc[:, col_a], df_integer_columns.loc[:, col_b])
    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']
    return(result)


# Checking the variance of columns
def column_variance(df):
    if type(df).__name__ != "DataFrame":
        raise Exception("Sorry, it is not dataframe!")
    return df.var()