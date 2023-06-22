import streamlit as st
from PIL import Image
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.style
from functools import partial
import mne
import pickle
from sklearn.preprocessing import LabelEncoder 
st.set_page_config(
    page_title="Brainwave Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
model = pickle.load(open("E:/Brainwave Analysis/project/model.pkl", "rb"))

# custom_theme = """
# [theme]
# primaryColor="#4a837e"
# backgroundColor="#dfecef"
# secondaryBackgroundColor="#7f98c9"
# textColor="#292e48"
# font="serif"
    
# """
# hide_default_format = """
#        <style>
      
#        footer {visibility: hidden;}
#        </style>
#        """


# Your Streamlit app code goes here...

# st.markdown(custom_theme, unsafe_allow_html=True)
df = pd.read_csv('https://raw.githubusercontent.com/mubeen161/Datasets/main/EEG.machinelearing_data_BRMH.csv')

# Set page configurations
df = df.rename({'sex': 'gender', 'eeg.date': 'eeg date', 'main.disorder': 'main disorder',
                'specific.disorder': 'specific disorder'}, axis=1)
df['age'] = df['age'].round(decimals=0)
df1=df.loc[:,'gender':'specific disorder']
df1=df1.drop('eeg date',axis=1)

def plot_eeg(levels, positions, axes, fig, ch_names=None, cmap='Spectral_r', cb_pos=(0.9, 0.1),cb_width=0.04, cb_height=0.9, marker=None, marker_style=None, vmin=None, vmax=None, **kwargs):
  if 'mask' not in kwargs:
    mask = np.ones(levels.shape[0], dtype='bool')
  else:
    mask = None
  im, cm = mne.viz.plot_topomap(levels, positions, axes=axes, names=ch_names,cmap=cmap, mask=mask, mask_params=marker_style, show=False, **kwargs)
def reformat_name(name):
    splitted = name.split(sep='.')
    if len(splitted) < 5:
        return name
    if splitted[0] != 'COH':
        result = f'{splitted[2]}.{splitted[4]}'
    else:
        result = f'{splitted[0]}.{splitted[2]}.{splitted[4]}.{splitted[6]}'
    return result


df.rename(reformat_name, axis=1, inplace=True)
# st.set_page_config(page_title='Streamlit Dashboard')


# Mean powers per main disorder
main_mean = df.groupby('main disorder').mean().reset_index()
# Mean powers per specific disorder
spec_mean = df.groupby('specific disorder').mean().reset_index()
# List of bands
msd=['Mood disorder','Addictive disorder','Trauma and stress related disorder','Schizophrenia','Anxiety disorder','Healthy control','Obsessive compulsive disorder']
bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
ssd=['Acute stress disorder','Adjustment disorder','Alcohol use disorder','Behavioral addiction disorder','Bipolar disorder','Depressive disorder','Healthy Control','Obsessive compulsive disorder','Panic disorder','Posttraumatic stress disorder','Schizophrenia','Social anxiety disorder']
# Convert from wide to long
main_mean = pd.wide_to_long(main_mean, bands, ['main disorder'], 'channel', sep='.', suffix='\w+')
spec_mean = pd.wide_to_long(spec_mean, bands, ['specific disorder'], 'channel', sep='.', suffix='\w+')

# Define channels
chs = {'FP1': [-0.03, 0.08], 'FP2': [0.03, 0.08], 'F7': [-0.073, 0.047], 'F3': [-0.04, 0.041],
       'Fz': [0, 0.038], 'F4': [0.04, 0.041], 'F8': [0.073, 0.047], 'T3': [-0.085, 0], 'C3': [-0.045, 0],
       'Cz': [0, 0], 'C4': [0.045, 0], 'T4': [0.085, 0], 'T5': [-0.073, -0.047], 'P3': [-0.04, -0.041],
       'Pz': [0, -0.038], 'P4': [0.04, -0.041], 'T6': [0.07, -0.047], 'O1': [-0.03, -0.08], 'O2': [0.03, -0.08]}
channels = pd.DataFrame(chs).transpose()
# Create a Streamlit app
# Define home page content
def home_page():
    st.title('MUFFAKHAM JAH COLLEGE OF ENGINEERING AND TECHNOLOGY')
    # st.write('This is the home page.')

    # Random text
    st.header('Computer Science and Artificial Intelligence Department')
    # random_text = [
    #     'Lorem ipsum dolor sit amet, consectetur adipiscing elit.',
    #     'Sed ut perspiciatis unde omnis iste natus error sit voluptatem.',
    #     'Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis.'
    # ]
    # st.write(random.choice(random_text))
    st.write('Brain Wave Prediction and Analytics')
    
    # Display images
    image_paths = ['E:/Brainwave Analysis/project/eeg.webp', 'E:/Brainwave Analysis/project/wave.webp', 'E:/Brainwave Analysis/project/process.jpg', 'E:/Brainwave Analysis/project/density.png','E:/Brainwave Analysis/project/pie.png','E:/Brainwave Analysis/project/bar.png','E:/Brainwave Analysis/project/chart.png','E:/Brainwave Analysis/project/future.png']
    # List of captions for the images
    captions = ['EEG Recording', 'Types of Waves', 'Process of EEG to ML', 'Density Plot','Main Disorder','Specific Disorder','Male-Famale Ratio','Future Scope']
    # Define the number of columns in the grid
    num_columns = 2
    # Calculate the number of rows based on the number of images and columns
    num_rows = len(image_paths) // num_columns
    # Loop over the rows and columns to display the images in a grid
    for row in range(num_rows):
        col1, col2 = st.columns(num_columns)
        for col in [col1, col2]:
            if image_paths and captions:
                image_path = image_paths.pop(0)
                caption = captions.pop(0)
                image = Image.open(image_path)
                col.image(image, caption=caption, use_column_width=True)





def plot_wave_comparison(df, x_axis, y_axis):
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot x-axis and y-axis
    ax1.plot(df[x_axis])
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title(x_axis)

    ax2.plot(df[y_axis])
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title(y_axis)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)


# Define Prediction page content
def prediction_page():
    # st.title('Prediction')
    # Add code for prediction here
    data=pd.read_csv('https://raw.githubusercontent.com/mubeen161/Datasets/main/EEG.machinelearing_data_BRMH.csv')
    data=data.rename(columns={"specific.disorder": "sd", "main.disorder": "md"})
    data = data.fillna(0)

    # Preprocess the data
    md = LabelEncoder()
    data['md'] = md.fit_transform(data['md'])
    sex=LabelEncoder()
    data['sex'] = sex.fit_transform(data['sex'])
    data['sd'] = md.fit_transform(data['sd'])
    data = data.drop(['eeg.date', 'no.'], axis=1)
    data=data.round(4)
    X = data.drop('sd', axis=1)
    y = data['sd']
    X = X.round(3)
    X["age"] = X["age"].round(0)
    class_mapping = {
        0: "Acute Stress Disorder",
        1: "Adjustment Disorder",
        2: "Alchohol Use Disorder",
        3: "Bipolar Disorder",
        4: "Behavioral Addictive Disorder", 
        5: "Depressive Disorder", 
        6: "Healthy Control",
        7: "Obsessive Compulsive Disorder",
        8: "Panic Disorder",
        9: "Post Traumatic Stress Disorder",
        10: "Schizophrenia",
        11: "Social Anxiety Disorder",
        # Add more mappings as needed
    }
    selected_data = st.selectbox("Select a person from Dataset :", X.index)

    # Retrieve the selected row from X_test
    input_array = X.loc[selected_data].values

    # Make prediction on user input
    if st.button("Predict"):
        features = np.array(input_array).reshape(1, -1)
        prediction = model.predict(features)
        # t=prediction[0]
        # ls = class_mapping[prediction[0]]
        st.success(f"The Disorder is {prediction[0]}")


# Define Plots page content
def plots_page():
    st.title('Plots')
    num_variables = st.sidebar.selectbox("Select number of variables:", [1, 2, 3])

    # Add code for plots here
    if num_variables == 1:
        plot_type1 = st.sidebar.selectbox("Select plot type:", ["Swarmplot", "Histogram", "Bar Plot","Line Plot","Scatter Plot"])
        
        # Select a single variable
        variable = st.sidebar.selectbox("Select a variable:", df1.columns)
        fig, ax = plt.subplots()
        if plot_type1 == "Swarmplot":
            sns.swarmplot(df1[variable])
            ax.set_xlabel(variable)
            # ax.set_ylabel("Index")
            ax.set_title(f"Swarm plot of {variable}")
        
        elif plot_type1 == "Histogram":
            ax.hist(df1[variable])
            ax.set_xlabel(variable)
            # ax.set_ylabel("Index")
            ax.set_title(f"Histogram of {variable}")
        
        elif plot_type1 == "Bar Plot":
            ax.bar(df1.index, df1[variable])
            ax.set_xlabel("Index")
            ax.set_ylabel(variable)
            ax.set_title(f"Bar plot of {variable}")
        elif plot_type1=="Line Plot":
            ax.plot(df1.index, df1[variable])
            plt.xlabel("Index")
            plt.ylabel(variable)
            plt.title("Line Plot of " + variable)
            plt.show()
        elif plot_type1=="Scatter Plot":
            fig, ax = plt.subplots()
            ax.scatter(df1.index, df1[variable])
            ax.set_xlabel("Index")
            ax.set_ylabel(variable)
            ax.set_title(f"Scatter Plot of {variable}")
        elif plot_type1 == "Violin Plot":
            ax.violinplot(df1[variable])
            ax.set_xlabel(variable)
            # ax.set_ylabel(y_variable)
            ax.set_title(f"Violin plot of {variable}")
    
        # elif plot_type1 == "Pair Plot":
        #     sns.pairplot(df1[variable])
        #     ax.set_xlabel(variable)
        #     # ax.set_ylabel(y_variable)
        #     ax.set_title(f"Pair plot of {variable} ")
        
        st.pyplot(fig)
    

    elif num_variables == 2:
        # Select two variables
        plot_type = st.sidebar.selectbox("Select plot type:", ["Bar Plot","Line Plot","Scatter Plot","Violin Plot","Kde Plot","Hexbin Plot","Area Plot"])
        x_variable = st.sidebar.selectbox("Select the x-axis variable:", df1.columns)
        y_variable = st.sidebar.selectbox("Select the y-axis variable:", df1.columns)
        fig, ax = plt.subplots()
    
        if plot_type == "Scatter Plot":
            ax.scatter(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Scatter plot of {x_variable} vs {y_variable}")
        
        elif plot_type == "Line Plot":
            ax.plot(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Line plot of {x_variable} vs {y_variable}")
        
        elif plot_type == "Bar Plot":
            ax.bar(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Bar plot of {x_variable} vs {y_variable}")
    
    
        elif plot_type == "Violin Plot":
            sns.violinplot(x=df1[x_variable], y=df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Violin plot of {x_variable} and {y_variable}")


    
        elif plot_type == "Kde Plot":
            sns.kdeplot(data=df1, x=df1[x_variable], y=df1[y_variable], shade=True)
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Kde plot of {x_variable} and {y_variable}")
        elif plot_type == "Hexbin Plot":
            ax.hexbin(df1[x_variable], df1[y_variable], gridsize=20, cmap='viridis')
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Hexbin plot of {x_variable} and {y_variable}")
        elif plot_type == "Area Plot":
            ax.fill_between(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Area plot of {x_variable} and {y_variable}")
    
        st.pyplot(fig)
    elif num_variables == 3:
        # Select three variables
        plot_type = st.sidebar.selectbox("Select plot type:", ["Scatter Plot", "Bar Plot", "Line Plot"])
        x_variable = st.sidebar.selectbox("Select the x-axis variable:", df.columns)
        y_variable = st.sidebar.selectbox("Select the y-axis variable:", df.columns)
        z_variable = st.sidebar.selectbox("Select the z-axis variable:", df.columns)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    
        if plot_type == "Scatter Plot":
            ax.scatter3D(df[x_variable], df[y_variable], df[z_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_zlabel(z_variable)
            ax.set_title(f"Scatter plot of {x_variable}, {y_variable}, {z_variable}")
    
    
        elif plot_type == "Bar Plot":
            ax.bar3d(df[x_variable], df[y_variable], 0, 1, 1, df[z_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_zlabel(z_variable)
            ax.set_title(f"Bar plot of {x_variable}, {y_variable}, {z_variable}")
    
        elif plot_type == "Line Plot":
            ax.plot3D(df[x_variable], df[y_variable], df[z_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_zlabel(z_variable)
            ax.set_title(f"Line plot of {x_variable}, {y_variable}, {z_variable}")
    
    
        
        st.pyplot(fig)

# Define Wave Compare page content
def wave_compare_page():
    st.title('Wave Comparison')
    image_path = "E:/Brainwave Analysis/project/head.png"
    
    # Open the image file using PIL
    image = Image.open(image_path)
    resized_image = image.resize((8, 8))
    
    # Calculate the center alignment
    image_width, image_height = resized_image.size
    
    # Display the resized image with center alignment
    st.sidebar.image(image, width=image_width, caption='Electrode Position Image', use_column_width=True, output_format='PNG')

    df1=df.drop(['no.','gender','age','eeg date','education','IQ','main disorder','specific disorder'],axis=1)
    x_axis = st.selectbox('Select Wave 1 : ', df1.columns)
    y_axis = st.selectbox('Select Wave 2 : ', df1.columns)

    # Check if the DataFrame is not empty
    if not df.empty:
        # Call the function to plot the wave comparison
        plot_wave_comparison(df, x_axis, y_axis)
    else:
        st.write('No data available to plot.')

def topographic_brain_activity():
    # st.write("Topographic Brain Activity selected")
    # Add your code for Topographic Brain Activity functionality here
    st.title("EEG Data Analysis")
    st.title('Brain Compare')
    img_pt='E:/Brainwave Analysis/project/level.png'
    # Add code for brain compare here
    image = Image.open(img_pt)
    image_width, image_height = image.size
    
    # Display the resized image with center alignment
    st.sidebar.image(image, width=image_width, caption='Brainwave Intensity Representation', use_column_width=True, output_format='PNG')
    test = spec_mean.loc[st.selectbox("Disorder",ssd), st.selectbox("Bnads",bands)]
    # Display the EEG plot
    fig, ax = plt.subplots()
    plot_eeg(test, channels.to_numpy(), ax, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    st.pyplot(fig)

def disorder_comparison():
    st.write("Disorder Comparison selected")
    # Add your code for Disorder Comparison functionality here
    test_schizo = main_mean.loc[st.selectbox("Disorder 1",msd), st.selectbox("Bnads 1",bands)]
    test_control = main_mean.loc[st.selectbox("Disorder 2",msd), st.selectbox("Bnads 2",bands)]
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
   
    # Plot the first subplot
    plot_eeg(test_schizo, channels.to_numpy(), ax1, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    ax1.set_title('Trauma and stress related disorder')
   
    # Plot the second subplot
    plot_eeg(test_control, channels.to_numpy(), ax2, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    ax2.set_title('Healthy control')
   
    # Display the plot
    st.pyplot(fig)

def brain_simulation():
    st.write("Brain Simulation selected")
    gif_path = "E:/Brainwave Analysis/project/plot.gif"
    img=Image.open(gif_path)
    # Display the GIF image
    st.image(img, use_column_width=True)
# Define Brain Compare page content
def brain_compare_page():
    st.title('Brain Compare')
    img_pt='E:/Brainwave Analysis/project/level.png'
    # Add code for brain compare here
    image = Image.open(img_pt)
    image_width, image_height = image.size
    
    # Display the resized image with center alignment
    st.sidebar.image(image, width=image_width, caption='Brainwave Intensity Representation', use_column_width=True, output_format='PNG')
    if st.button("Topographic Brain Activity"):
        topographic_brain_activity()

    if st.button("Disorder Comparison"):
        disorder_comparison()

    if st.button("Brain Simulation"):
        brain_simulation()


# Define Stress Level page content
def stress_level_page():
    st.title('Stress Level')
    # Add code for stress level here
    column_name = st.selectbox("Select Band",bands)
    highest_value = main_mean[column_name].max()
    lowest_value = main_mean[column_name].min()
    

    min_value = round(lowest_value, 2)
    max_value = round(highest_value, 2)


    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_axis_off()


    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(min_value, max_value)


    ax.imshow(np.arange(min_value, max_value).reshape(1, -1), cmap=cmap, norm=norm, aspect='auto')
    ax.text(0, 0, str(min_value), ha='left', va='center', color='black', weight='light')
    ax.text(max_value - min_value, 0, str(max_value), ha='right', va='center', color='black', weight='light')
    plt.title('Stress level using beta brain waves')
    st.pyplot(fig)
    # st.write("Brain Simulation selected")
    img_path = "E:/Brainwave Analysis/project/stresslevel.jpg"
    img=Image.open(img_path)
    # Display the GIF image
    st.image(img,caption='Reference Chart' ,use_column_width=True)

    # Display the plot using Streamlit
    


# Main code
def main():
    # Dropdown menu for page selection
    page_options = {
        'Home': home_page,
        'Prediction': prediction_page,
        'Plots': plots_page,
        'Wave Compare': wave_compare_page,
        # 'Brain Compare': brain_compare_page,
        'Stress Level': stress_level_page,
        'Topographic Brain Activity':topographic_brain_activity,
        'Disorder Comparison':disorder_comparison,
        'Brain Simulation':brain_simulation
    }
    selected_page = st.sidebar.selectbox('Select a page', list(page_options.keys()))
    page = page_options[selected_page]

    # Display selected page content
    page()


if __name__ == '__main__':
    main()
