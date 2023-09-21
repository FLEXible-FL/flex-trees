import pandas as pd

def preprocess_adult(training_dataset):
    """Function to preprocess Adult dataset and transform it to a categorical dataset.
    Function assumes dataset has columns labels ['x0', 'x1', 'x2',...,'x13', 'label']
    The feature 'fnlwgt' ('x2') will be dropped if it is in the dataset.

    Args:
        training_dataset (pd.DataFrame): Adult DataFrame

    Returns:
        pd.DataFrame: Adult transformed into a categorical dataset.
    """
    if 'x2' in training_dataset.columns:
        training_dataset = training_dataset.drop(['x2'], axis=1)
    training_dataset['x0'] = training_dataset['x0'].astype(int)
    training_dataset['x4'] = training_dataset['x4'].astype(int)
    training_dataset['x10'] = training_dataset['x10'].astype(int)
    training_dataset['x11'] = training_dataset['x11'].astype(int)
    training_dataset['x12'] = training_dataset['x12'].astype(int)
    
    age = training_dataset['x0']
    edu_num = training_dataset['x4']
    cap_gain = training_dataset['x10']
    cap_loss = training_dataset['x11']
    hours = training_dataset['x12']
    
    age_bins = [0, 18, 40, 80, 99]
    age_labels = ['teen', 'adult', 'old-adult', 'elder']
    edu_num_bins = [0, 5, 10, 17]
    edu_num_labels = ['<5', '5-10', '>10']
    cap_gain_bins = [-1, 1, 39999, 49999, 79999, 99999]
    cap_gain_labels = [0, 1, 2, 3, 4]
    cap_loss_bins = [-1, 1, 999, 1999, 2999, 3999, 4499]
    cap_loss_labels = [0, 1, 2, 3, 4, 5]
    hr_bins = [0, 20, 40, 100]
    hr_labels = ['<20', '20-40', '>40']

    training_dataset['x0'] = pd.cut(age, bins=age_bins, labels=age_labels)
    training_dataset['x4'] = pd.cut(edu_num, bins=edu_num_bins, labels=edu_num_labels)
    training_dataset['x10'] = pd.cut(cap_gain, bins=cap_gain_bins, labels=cap_gain_labels)
    training_dataset['x11'] = pd.cut(cap_loss, bins=cap_loss_bins, labels=cap_loss_labels)
    training_dataset['x12'] = pd.cut(hours, bins=hr_bins, labels=hr_labels)

    training_dataset['x1'] = training_dataset['x1'].map({
        " ?": 'others',
        " Federal-gov": 'gov', " Local-gov": 'gov',
        " Never-worked":'others',
        " Private": 'others',
        " Self-emp-inc": 'self',
        " Self-emp-not-inc":'self', " State-gov": 'gov',
        " Without-pay": 'others'
    })

    training_dataset['x3'] = training_dataset['x3'].map({
        " 10th": 'non_college'," 11th": 'non_college',
        " 12th": 'non_college', " 1st-4th": 'non_college',
        " 5th-6th": 'non_college', " 7th-8th": 'non_college',
        " 9th": 'non_college',
        " Assoc-acdm": 'assoc' , " Assoc-voc": 'assoc',
        " Bachelors": 'college',
        " Doctorate": 'grad', " HS-grad": 'grad', " Masters": 'grad',
        " Preschool": 'others',
        " Prof-school": 'others',
        " Some-college": 'college'
    })

    training_dataset['x13'] = training_dataset['x13'].map({
        ' ?':'others',
        ' Cambodia': 'asia',
        ' Canada': 'north_america',
        ' China': 'asia',
        ' Columbia': 'south_america',
        ' Cuba': 'south_america',
        ' Dominican-Republic': 'south_america',
        ' Ecuador': 'south_america',
        ' El-Salvador': 'south_america',
        ' England': 'europe',
        ' France': 'europe',
        ' Germany':'europe',
        ' Greece':'europe',
        ' Guatemala': 'south_america',
        ' Haiti': 'south_america',
        ' Holand-Netherlands': 'europe',
        ' Honduras': 'south_america' ,
        ' Hong': 'asia',
        ' Hungary':'europe',
        ' India':'asia',
        ' Iran': 'asia',
        ' Ireland': 'europe',
        ' Italy': 'europe',
        ' Jamaica': 'south_america',
        ' Japan': 'asia',
        ' Laos': 'asia',
        ' Mexico': 'south_america',
        ' Nicaragua': 'south_america',
        ' Outlying-US(Guam-USVI-etc)': 'north_america',
        ' Peru': 'south_america',
        ' Philippines': 'asia',
        ' Poland': 'europe',
        ' Portugal': 'europe',
        ' Puerto-Rico': 'south_america',
        ' Scotland':'europe' ,
        ' South': 'asia',
        ' Taiwan': 'asia',
        ' Thailand': 'asia',
        ' Trinadad&Tobago': 'south_america',
        ' United-States': 'north_america',
        ' Vietnam': 'asia', ' Yugoslavia':'europe'
    })

    training_dataset['x0'] = training_dataset['x0'].astype('category')
    training_dataset['x1'] = training_dataset['x1'].astype('category')
    training_dataset['x3'] = training_dataset['x3'].astype('category')
    training_dataset['x4'] = training_dataset['x4'].astype('category')
    training_dataset['x10'] = training_dataset['x10'].astype('category')
    training_dataset['x11'] = training_dataset['x11'].astype('category')
    training_dataset['x12'] = training_dataset['x12'].astype('category')
    training_dataset['x13'] = training_dataset['x13'].astype('category')

    # training_dataset['label'] = training_dataset.apply(lambda row: 1 if '>50K' in row['label'] else 0, axis=1)

    return training_dataset

def preprocess_credit2(training_dataset):
    """Function to preprocess Credit2 dataset and transform it to a categorical dataset.
    Function assumes dataset has columns labels ['x1', 'x1', 'x2',...,'x22']

    Args:
        training_dataset (pd.DataFrame): Credit2 DataFrame

    Returns:
        pd.DataFrame: Credit2 transformed into a categorical dataset.
    """
    # Amount of the given credit -> To deciles
    training_dataset['X1'] = pd.cut(training_dataset['X1'], bins=10, labels=[
        "(9999.999, 30000.0]", "(100000.0, 140000.0]", "(70000.0, 100000.0]",
        "(30000.0, 50000.0]", "(360000.0, 1000000.0]", "(180000.0, 210000.0]",
        "(210000.0, 270000.0]", "(50000.0, 70000.0]", "(270000.0, 360000.0]",
        "(140000.0, 180000.0]"
    ])
    # Gender
    training_dataset['X2'] = training_dataset['X2'].map({
        1:'male',
        2:'female'
    })
    # Education
    training_dataset['X3'] = training_dataset['X3'].map({
        1:'graduate_school',
        2:'university',
        3:'high_school',
        4:'others',
        0:'others',
        5:'others',
        6:'others'
    })
    # Marital status
    training_dataset['X4'] = training_dataset['X4'].map({
        1:'married',
        2:'single',
        3:'others',
        0:'others'
    })
    # Age
    age_bins = [0, 18, 40, 80, 99]
    age_labels = ['teen', 'adult', 'old-adult', 'elder']
    training_dataset['X5'] = pd.cut(training_dataset['X5'], bins=age_bins,
                                    labels=age_labels)
    training_dataset['X5'] = training_dataset['X5'].astype('category')
    # History of past payments (April-September 2015): X6-X11.
    # Features -2, and 0 were set with random names
    training_dataset['X6'] = training_dataset['X6'].map({
        -2:'pay_duty_t',
        -1:'pay_duty',
        0:'payed',
        1:'pay_delay_one_month',
        2:'pay_delay_two_months',
        3:'pay_delay_three_months',
        4:'pay_delay_four_months',
        5:'pay_delay_five_months',
        6:'pay_delay_six_months',
        7:'pay_delay_seven_months',
        8:'pay_delay_eight_months',
        9:'pay_delay_nine_months_above'
    })
    training_dataset['X7'] = training_dataset['X7'].map({
        -2:'pay_duty_t',
        -1:'pay_duty',
        1:'pay_delay_one_month',
        2:'pay_delay_two_months',
        3:'pay_delay_three_months',
        4:'pay_delay_four_months',
        5:'pay_delay_five_months',
        6:'pay_delay_six_months',
        7:'pay_delay_seven_months',
        8:'pay_delay_eight_months',
        9:'pay_delay_nine_months_above',
        0:'payed'
    })
    training_dataset['X8'] = training_dataset['X8'].map({
        -2:'pay_duty_t',
        -1:'pay_duty',
        1:'pay_delay_one_month',
        2:'pay_delay_two_months',
        3:'pay_delay_three_months',
        4:'pay_delay_four_months',
        5:'pay_delay_five_months',
        6:'pay_delay_six_months',
        7:'pay_delay_seven_months',
        8:'pay_delay_eight_months',
        9:'pay_delay_nine_months_above',
        0:'payed'
    })
    training_dataset['X9'] = training_dataset['X9'].map({
        -2:'pay_duty_t',
        -1:'pay_duty',
        1:'pay_delay_one_month',
        2:'pay_delay_two_months',
        3:'pay_delay_three_months',
        4:'pay_delay_four_months',
        5:'pay_delay_five_months',
        6:'pay_delay_six_months',
        7:'pay_delay_seven_months',
        8:'pay_delay_eight_months',
        9:'pay_delay_nine_months_above',
        0:'payed'
    })
    training_dataset['X10'] = training_dataset['X10'].map({
        -2:'pay_duty_t',
        -1:'pay_duty',
        1:'pay_delay_one_month',
        2:'pay_delay_two_months',
        3:'pay_delay_three_months',
        4:'pay_delay_four_months',
        5:'pay_delay_five_months',
        6:'pay_delay_six_months',
        7:'pay_delay_seven_months',
        8:'pay_delay_eight_months',
        9:'pay_delay_nine_months_above',
        0:'payed'
    })
    training_dataset['X11'] = training_dataset['X11'].map({
        -2:'pay_duty_t',
        -1:'pay_duty',
        1:'pay_delay_one_month',
        2:'pay_delay_two_months',
        3:'pay_delay_three_months',
        4:'pay_delay_four_months',
        5:'pay_delay_five_months',
        6:'pay_delay_six_months',
        7:'pay_delay_seven_months',
        8:'pay_delay_eight_months',
        9:'pay_delay_nine_months_above',
        0:'payed'
    })
    # Amount of bill statement (NT dollar) in September 2005-August2005-...-April2005
    training_dataset['X12']= pd.qcut(training_dataset['X12'], [0, .25, .5, .75, 1], labels=[ 
        '(-165580.001, 3558.75]', '(3558.75, 22381.5]', '(22381.5, 67091.0]', '(67091.0, 964511.0]'
        ]
    )
    training_dataset['X13']= pd.qcut(training_dataset['X13'], [0, .25, .5, .75, 1], labels=[ 
        '(-157264.001, 2666.25]', '(2666.25, 20088.5]', '(20088.5, 60164.75]', '(60164.75, 1664089.0]'
        ]
    )
    training_dataset['X14']= pd.qcut(training_dataset['X14'], [0, .25, .5, .75, 1], labels=[ 
        '(-157264.001, 2666.25]', '(2666.25, 20088.5]', '20088.5, 60164.75]', '(60164.75, 1664089.0]'
        ]
    )
    training_dataset['X15']= pd.qcut(training_dataset['X15'], [0, .25, .5, .75, 1], labels=[ 
        '(-170000.001, 2326.75]', '(2326.75, 19052.0]', '19052.0, 54506.0]', '(54506.0, 891586.0]'
        ]
    )
    training_dataset['X16']= pd.qcut(training_dataset['X16'], [0, .25, .5, .75, 1], labels=[ 
        '(-81334.001, 1763.0]', '(1763.0, 18104.5]', '(18104.5, 50190.5]', '50190.5, 927171.0]'
        ]
    )
    training_dataset['X17']= pd.qcut(training_dataset['X17'], [0, .25, .5, .75, 1], labels=[ 
        '(-339603.001, 1256.0]', '(1256.0, 17071.0]', '17071.0, 49198.25]', '(49198.25, 961664.0]'
        ]
    )
    # Amount of previous payment(NT dollar) in September 2005-...-April 2005
    training_dataset['X18']= pd.qcut(training_dataset['X18'], [0, .25, .5, .75, 1], labels=[ 
        '(-0.001, 1000.0]', '(1000.0, 2100.0]', '(2100.0, 5006.0]', '5006.0, 873552.0]'
        ]
    )
    training_dataset['X19']= pd.qcut(training_dataset['X19'], [0, .25, .5, .75, 1], labels=[ 
        '(-0.001, 833.0]', '(833.0, 2009.0]', '(2009.0, 5000.0]', '5000.0, 1684259.0]'
        ]
    )
    training_dataset['X20']= pd.qcut(training_dataset['X20'], [0, .25, .5, .75, 1], labels=[ 
        '(-0.001, 390.0]', '(390.0, 1800.0]', '(1800.0, 4505.0]', '4505.0, 896040.0]'
        ]
    )
    training_dataset['X21']= pd.qcut(training_dataset['X21'], [0, .25, .5, .75, 1], labels=[ 
        '(-0.001, 296.0]', '(296.0, 1500.0]', '(1500.0, 4013.25]', '4013.25, 621000.0]'
        ]
    )
    training_dataset['X22']= pd.qcut(training_dataset['X22'], [0, .25, .5, .75, 1], labels=[ 
        '(-0.001, 252.5]', '(252.5, 1500.0]', '(1500.0, 4031.5]', '4031.5, 426529.0]'
        ]
    )
    training_dataset['X23']= pd.qcut(training_dataset['X23'], [0, .25, .5, .75, 1], labels=[ 
        '(-0.001, 117.75]',  '(117.75, 1500.0]',  '(1500.0, 4000.0]', '4000.0, 528666.0]'
        ]
    )
    return training_dataset


