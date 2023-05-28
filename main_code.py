import numpy as np
import video_process
import pandas as pd
import os

# database address
input = "./input"

lens=len(os.listdir(input))
video_names=np.array(os.listdir(input))
feature=np.zeros((lens,19))
i=0

for i,video_name in enumerate(os.listdir(input)):
    print(i)
    feature[i,:]=video_process.process(os.path.join(input, video_name))
    i = i + 1

video_names_and_feature=np.hstack((video_names.reshape(lens,1),feature))
Final_csv = pd.DataFrame(np.array(video_names_and_feature))
Final_csv.to_csv('./output/submission.csv', sep=',', index=False, header=False)

print("operation completed successfully")

