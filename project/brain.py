import streamlit as st
import mne
import matplotlib.pyplot as plt

def generate_montage(chs):
    mont = mne.channels.make_dig_montage(chs)
    fig = plt.figure()
    mont.plot()
    st.pyplot(fig)  # Display the plot in Streamlit
    plt.close(fig)  # Close the figure to free up memory
    
    # Save the figure and display it as an image in Streamlit
    fig.savefig('head.png', bbox_inches='tight', transparent=True)
    st.image('head.png', use_column_width=True)

def main():
    st.title("Montage Visualization")
    
    # Define your channel information
    chs = {
        'FP1': [-0.03, 0.08],
        'FP2': [0.03, 0.08],
        'F7': [-0.073, 0.047],
        'F3': [-0.04, 0.041],
        'Fz': [0, 0.038],
        'F4': [0.04, 0.041],
        'F8': [0.073, 0.047],
        'T3': [-0.085, 0],
        'C3': [-0.045, 0],
        'Cz': [0, 0],
        'C4': [0.045, 0],
        'T4': [0.085, 0],
        'T5': [-0.073, -0.047],
        'P3': [-0.04, -0.041],
        'Pz': [0, -0.038],
        'P4': [0.04, -0.041],
        'T6': [0.07, -0.047],
        'O1': [-0.03, -0.08],
        'O2': [0.03, -0.08]
    }
    
    generate_montage(chs)
    
if __name__ == '__main__':
    main()


