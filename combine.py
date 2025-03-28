from fileinput import filename
import glob
from re import A
from pydub import AudioSegment
import sys
import glob
import os

def combine(filename,name):
    #filename1 = './result'+name+'/'+filename+'.wav'
    filename1='./results/'+name+'/'+filename
    # filename1='/data/yiwenl/zgrsingle/CoMoSVC/result_teacher/不如2.wav'
    # filename1='/data/yiwenl/SVC_BigModel/spk_embedding/EDM_WAVLM_INFER/results/zsw_song/一生何求男.wav'

    print(filename1)
    filename2='/data/yiwenl/SVC_BigModel/spk_embedding/EDMSVC/accompaniment/'+filename
    # filename2='./accompaniment/一生何求男.wav'

    print(filename2)
    track1 = AudioSegment.from_wav(filename1)
    # track1=track1+3
    track2 = AudioSegment.from_wav(filename2)
    # track2=track2-5    
    time1=track1.duration_seconds
    time2=track2.duration_seconds
    time=min(time1,time2)
    if time1<time2:
        track2=track2[:time*1000]
    else:
        track1=track1[:time*1000]
    print(type(track1))
    #output = AudioSegment.from_mono_audiosegments(sound1, sound2)
    #combined = AudioSegment.from_mono_audiosegments(track1, track2)
    combined=track1.overlay(track2)
    combined.export('./finals/'+name+"/"+filename, format="wav")


if __name__ == '__main__':
    singer=sys.argv[1]
    num=len(sys.argv)
    filenames=[]
    for i in range(2,num):
        filenames.append(sys.argv[i])
    print(filenames)

    for file in filenames:
        combine(file,singer)