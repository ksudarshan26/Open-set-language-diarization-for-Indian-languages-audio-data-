import numpy as np
import math
import os,cv2
import sys
import wave
import contextlib as cl
import re
import shutil
import time
print("Hello User\nTeam HunTer_SQuaD Welcomes You")
import multiprocessing as m
import soundfile as sf
import concurrent.futures
import speech_recognition as sr
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.silence import split_on_silence
from collections import Counter as cn
from googlesearch import search




def noise_reduce(audio_data,rate,win_length,amp_adjust):
    print("Analyzing the audio")
    import noisereduce as nr
    
    mean=0
    noise_list=[]
    mean_list=[]
    win_length=win_length
    amp_adjust=amp_adjust
    for j in range(0,len(audio_data)):
        if(j>0 and j%win_length==0):   
            mean=math.sqrt(mean/win_length)
            
            mean_list.append(mean)
            mean=0
            mean+=audio_data[j]**2
        else:
            mean+=audio_data[j]**2
    k=0
    
    for i in range(0,len(mean_list)):
        if i>0 and i<len(mean_list)-1:
            if(mean_list[i-1]<mean_list[i]+amp_adjust and mean_list[i-1]>mean_list[i]-amp_adjust):
                if(mean_list[i+1]<mean_list[i]+amp_adjust and mean_list[i+1]>mean_list[i]-amp_adjust):
                    if k==0:
                        noise_list.append((i-1)*win_length)
                        k+=1
                else:
                    if(k>0):
                        noise_list.append((i+1)*win_length)
                        k=0
                    else:
                        k=0
                        noise_list.append((i-1)*win_length)
                        noise_list.append((i+1)*win_length)
        elif(i==len(mean_list)-1):
            if(mean_list[i-1]<mean_list[i]+amp_adjust and mean_list[i-1]>mean_list[i]-amp_adjust):
                noise_list.append((i+1)*win_length)
                

    diff=0
    for i in range(0,len(noise_list)-1):
        if(i%2==0):
            if(noise_list[i+1]-noise_list[i] > diff):
                diff = noise_list[i+1]-noise_list[i]
                noise_start = noise_list[i]+win_length
                noise_stop = noise_list[i+1]-win_length
    noisy_part =audio_data[noise_start:noise_stop]
    # perform noise reduction
    nr_data = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part,n_grad_freq=2,n_grad_time=16,n_fft=2048,win_length=2048,hop_length=512,n_std_thresh=1.5,prop_decrease=1.0)
    nr_data = np.int16(nr_data/np.max(np.abs(nr_data))*32768)
    write('test212.wav', rate,nr_data[512:len(nr_data)-512])
    print("Analyzing Succesfully Completed")
    print("Recognizing started")


def segmentation(min_silence_len,silence_thresh,chunk_silent):
    
    song = AudioSegment.from_file('test212.wav',format="wav")
    #print(song.frame_rate)
    shutil.rmtree("audio_chunks")
    chunk_dur=[]
    #print(4)
    chunks,silent_ranges= split_on_silence(song,min_silence_len = min_silence_len,silence_thresh = silence_thresh) 
    #print(6)
    try: 
        os.mkdir('audio_chunks') 
        
    except(FileExistsError): 
        pass  
    i=0
    chunk_silent = AudioSegment.silent(duration = 100)
    for chunk in chunks:
        audio_chunk = chunk_silent + chunk + chunk_silent 
        audio_chunk.export("./audio_chunks/chunk{0}.wav".format(i), bitrate ='64k', format ="wav")
        i+=1
    filename = './audio_chunks/chunk'+str(i-1)+'.wav'
    with cl.closing(wave.open(filename,'r')) as f:
        rate1 = f.getframerate()
    chunk_rate = rate1
    nf_chunks=i
    return chunk_rate,nf_chunks,silent_ranges




def recognizing_text(nf_chunks,recognized_text,):
    print("Recognizing The Audio")
    def google_requests(i):
        filename = './audio_chunks/chunk'+str(i)+'.wav'
        with cl.closing(wave.open(filename,'r')) as f:
            frames1 = f.getnframes()
            rate1 = f.getframerate()
            dur1 = (frames1/ float(rate1))
            dur1=dur1-0.4
        r = sr.Recognizer()
        
        
        with sr.AudioFile(filename) as source:  
            audio_listened = r.listen(source)
            
        try: 
            rec = r.recognize_google(audio_listened, language='en-IN' 
            rec=rec.lower()
            #txtfile=open(filename1,"w");txtfile.write(rec);txtfile.close()
            
        
        except sr.UnknownValueError: 
            #print("Could not understand audio")
            rec =' '
        
        except sr.RequestError as e:
            print("Could not request results. check your internet connection:" + e)
        return [i,rec,dur1]
    

    audio_list=os.listdir('audio_chunks')
    audio_list=audio_list[0:nf_chunks]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0,len(audio_list)):
            futures.append(executor.submit(google_requests,i))
        for future in concurrent.futures.as_completed(futures):
            recognized_text.append(future.result())
    print("Recognition Succesfully Completed")

            


def lang_correction(lang_out,dur_out):
    for i in range(0,len(lang_out)):
        if i==0:
            if dur_out[i][0]!=0:
                dur_out[i][0]=0
        elif i==len(lang_out)-1:
            if dur_out[i][1]!=dur:
                dur_out[i][1]=dur
        else:
            if dur_out[i+1][0] < dur_out[i][1]:
                dur_out[i][1]=dur_out[i+1][0]
            elif(dur_out[i+1][0] - dur_out[i][1])>5:
                dur_out[i][1]=dur_out[i+1][0]
    for i in range(0,len(lang_out)):
        if i==0:
            if lang_out[i]=='<mixed>':
                dur_out[i+1][0]=0
                dur_out[i]=[]
        else:
            if lang_out[i]=='<mixed>':
                dur_out[i-1][1]=dur_out[i][1]
                dur_out[i]=[]
    #print(lang_out,dur_out)
    lang_out1=[l for l in lang_out if l!='<mixed>']
    dur_out1=[l for l in dur_out if l!=[]]
    for i in range(0,len(lang_out1)):
        if i!=0:
            if(lang_out1[i]==lang_out1[i-1]):
                lang_out1[i]='<remove>'
                dur_out1[i-1][1]=dur_out1[i][1]
                dur_out1[i]=[]
    lang_out=[l for l in lang_out1 if l!='<remove>']
    dur_out=[l for l in dur_out1 if l!=[]]
    return lang_out,dur_out



def chunk_time(nf_chunks,recognized_text,silence_ranges):
    print("Language identification started")
    chunk_dur=[]
    for i in range(0,nf_chunks):
        chunk_dur.append(recognized_text[i][2])
    for i in range(0,nf_chunks):
        if(len(chunk_dur)==len(silence_ranges)):
            chunk_dur[i]+=(silence_ranges[i][1]-silence_ranges[i][0])/1000
        elif(len(chunk_dur) < len(silence_ranges)):
            if(i==0):
                chunk_dur[i]+=(silence_ranges[i][1]-silence_ranges[i][0])/1000 +(silence_ranges[i+1][1]-silence_ranges[i+1][0])/1000
            else:
                chunk_dur[i]+=(silence_ranges[i+1][1]-silence_ranges[i+1][0])/1000
        elif(len(chunk_dur) > len(silence_ranges)):
            if(i<len(silence_ranges)):
                chunk_dur[i]+=(silence_ranges[i][1]-silence_ranges[i][0])/1000
    return chunk_dur



global lang
def lang_files():
    
    lang=["telugu","tamil","hindi","english","bengali","kannada","mixed","unknown"]
    lang_dir={}
    for i in lang:
        L = [line.rstrip('\n') for line in open("./dataset/{0}20.txt".format(i),encoding="utf8")]
        #print(L)
        if len(L)!=0:
            lang_dir["{0}".format(i)]=L[0].split()
        else:
            lang_dir["{0}".format(i)]=[]
    return lang,lang_dir






def lang_classify(lang,lang_dir,nf_chunks,chunk_word_limit,recognized_text):
    print("Language Classification Started")
    mixed_list=[]
    cl_lang_and_index_dir={}
    for i in lang:
        cl_lang_and_index_dir["{0}".format(i)]=[]   
    
    def samp0(lang,chunk_dir,chunkindex_dir,lang_dir,cl_lang_and_index_dir):
        unknown_list=[]
        for i1 in chunk_dir:
            i2=-1
            for i in chunk_dir[i1]:
                i2+=1
                k=0
                for j in lang_dir:
                    if i in lang_dir["{0}".format(j)]:
                        k+=1
                        j1=j
                if k==1:
                    L=cl_lang_and_index_dir["{0}".format(j1)]
                    L.append([chunkindex_dir[i1][i2],i])
                    cl_lang_and_index_dir["{0}".format(j1)]=L
                
                elif k>1:
                    L=cl_lang_and_index_dir["mixed"]
                    L.append([chunkindex_dir[i1][i2],i])
                    cl_lang_and_index_dir["mixed"]=L                
                                                        
                else:
                    unknown_list.append([chunkindex_dir[i1][i2],i,"unknown"])
        #unknown_list_final=unknown_list


        unknown_list_final=unknown_list
        
        #print(9)
        unknown_list_final = sorted(unknown_list_final,key=lambda x:x[0])
        pop_list=[]
        #print(unknown_list_final)
        for unknown in range(0,len(unknown_list_final)):
            
            if unknown_list_final[unknown][2] != "unknown":
                filename=unknown_list_final[unknown][2]+'20.txt'
                #txtfile=open(filename,'a');txtfile.write(' '+unknown_list_final[unknown][1]);txtfile.close()
                #print(unknown_list_final[unknown][2])
                L=cl_lang_and_index_dir["{0}".format(unknown_list_final[unknown][2])]
                L.append(unknown_list_final[unknown][0:2])
                L = sorted(L,key=lambda x:x[0])
                cl_lang_and_index_dir["{0}".format(unknown_list_final[unknown][2])]=L
                pop_list.append(unknown)
        #print(pop_list)
        #print(len(pop_list))
        #print(unknown_list_final)
        for i in pop_list:
            unknown_list_final[i]=[]
        unknown_list_final=[l for l in unknown_list_final if len(l)>0]
        pop_list=[]
        
        for m in range(0,len(unknown_list_final)-2):
            if((unknown_list_final[m][0]+1)==unknown_list_final[m+1][0]):
                if((unknown_list_final[m+1][0]+1)==unknown_list_final[m+2][0]):
                    L=cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]
                    L.append(unknown_list_final[m][0:2])
                    L = sorted(L,key=lambda x:x[0])
                    cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]=L
                    pop_list.append(m)
                else:
                    if(chunkindex_dir["chunk{0}".format(math.floor(unknown_list_final[m+1][0]/chunk_word_limit)+1)]!=[]):
                        if(unknown_list_final[m+1][0]==chunkindex_dir["chunk{0}".format(math.floor(unknown_list_final[m+1][0]/chunk_word_limit))][-1]):
                            if(((math.floor(unknown_list_final[m+1][0]/chunk_word_limit)*chunk_word_limit) + chunk_word_limit)== unknown_list_final[m+2][0]):
                                L=cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]
                                L.append(unknown_list_final[m][0:2])
                                L = sorted(L,key=lambda x:x[0])
                                cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]=L
                                pop_list.append(m)

            else:
                if(chunkindex_dir["chunk{0}".format(math.floor(unknown_list_final[m][0]/chunk_word_limit)+1)]!=[]):
                    if(unknown_list_final[m][0]==chunkindex_dir["chunk{0}".format(math.floor(unknown_list_final[m][0]/chunk_word_limit))][-1]):
                        if(((math.floor(unknown_list_final[m][0]/chunk_word_limit)*chunk_word_limit) + chunk_word_limit)== unknown_list_final[m+1][0]):
                            if((unknown_list_final[m+1][0]+1)==unknown_list_final[m+2][0]):
                                L=cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]
                                L.append(unknown_list_final[m][0:2])
                                L = sorted(L,key=lambda x:x[0])
                                cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]=L
                                pop_list.append(m)
                            else:
                                if(chunkindex_dir["chunk{0}".format(math.floor(unknown_list_final[m+1][0]/chunk_word_limit)+2)]!=[]):
                                    if(unknown_list_final[m+1][0]==chunkindex_dir["chunk{0}".format(math.floor(unknown_list_final[m+1][0]/chunk_word_limit))][-1]):
                                        if(((math.floor(unknown_list_final[m+1][0]/chunk_word_limit)*chunk_word_limit) + chunk_word_limit)== unknown_list_final[m+2][0]):
                                            L=cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]
                                            L.append(unknown_list_final[m][0:2])
                                            L = sorted(L,key=lambda x:x[0])
                                            cl_lang_and_index_dir["{0}".format(unknown_list_final[m][2])]=L
                                            pop_list.append(m)
        #print(pop_list)
        #print(len(pop_list))
        for i in pop_list:
            unknown_list_final[i]=[]
        unknown_list_final=[l for l in unknown_list_final if len(l)>0]
        #print(unknown_list_final)
        
        for unknown in range(0,len(unknown_list_final)):
            if unknown_list_final[unknown][0]!=0:
                if unknown_list_final[unknown][0]%chunk_word_limit != 0:
                    for lang1 in lang:
                        L=cl_lang_and_index_dir["{0}".format(lang1)]
                        for unknown2 in range(0,len(L)):
                            if unknown_list_final[unknown][0]-1 == L[unknown2][0]:
                                L.append(unknown_list_final[unknown][0:2])
                                L = sorted(L,key=lambda x:x[0])
                                cl_lang_and_index_dir["{0}".format(lang1)]=L
                                
                else:
                    #print(chunkindex_dir["chunk{0}".format(int(unknown_list_final[unknown][0]/chunk_word_limit) - 1)])
                    unknown4 = 1
                    unknown6 = int(unknown_list_final[unknown][0]/chunk_word_limit) - 1
                    while unknown4 != 0:
                        if unknown6>=0:
                            unknown5 = chunkindex_dir["chunk{0}".format(unknown6)]
                        else:
                            break;
                        if unknown5 != []:
                            unknown3 = unknown5[-1]
                            unknown4 = 0
                        else:
                            if unknown6 <= 0:
                                unknown3='None'
                                break;
                            else:
                                unknown6-=1
                    if unknown3 != 'None':        
                        for lang1 in lang:
                            L=cl_lang_and_index_dir["{0}".format(lang1)]
                            for unknown2 in range(0,len(L)):
                                if unknown3==L[unknown2][0]:
                                    L.append(unknown_list_final[unknown][0:2])
                                    L = sorted(L,key=lambda x:x[0])
                                    cl_lang_and_index_dir["{0}".format(lang1)]=L
                    else:
                        L=cl_lang_and_index_dir["unknown"]
                        L.append(unknown_list_final[unknown][0:2])
                        L = sorted(L,key=lambda x:x[0])
                        cl_lang_and_index_dir["unknown"]=L

            elif(len(chunkindex_dir["chunk{0}".format(unknown_list_final[unknown][0])])!=1):
                 for lang1 in lang:
                    L=cl_lang_and_index_dir["{0}".format(lang1)]
                    for unknown2 in range(0,len(L)):
                        if unknown_list_final[unknown][0]+1==L[unknown2][0]:
                            L.append(unknown_list_final[unknown][0:2])
                            L = sorted(L,key=lambda x:x[0])
                            cl_lang_and_index_dir["{0}".format(lang1)]=L
            else:
                unknown4 = 1
                unknown6 = int(unknown_list_final[unknown][0]/chunk_word_limit) - 1
                while unknown4 != 0:
                    if unknown6>=0:
                        unknown5 = chunkindex_dir["chunk{0}".format(unknown6)]
                    else:
                        break;
                    unknown5 = chunkindex_dir["chunk{0}".format(unknown6)]
                    if unknown5 != []:
                        unknown3 = unknown5[0]
                        unknown4 = 0
                    else:
                        if unknown6 <= 0:
                            unknown3='None'
                            break;
                        else:
                            unknown6-=1
                if unknown3 != 'None':
                    for lang1 in lang:
                        L=cl_lang_and_index_dir["{0}".format(lang1)]
                        for unknown2 in range(0,len(L)):
                            if unknown3==L[unknown2][0]:
                                L.append(unknown_list_final[unknown][0:2])
                                L = sorted(L,key=lambda x:x[0])
                                cl_lang_and_index_dir["{0}".format(lang1)]=L
                else:
                    L=cl_lang_and_index_dir["unknown"]
                    L.append(unknown_list_final[unknown][0:2])
                    L = sorted(L,key=lambda x:x[0])
                    cl_lang_and_index_dir["unknown"]=L
    
    
    chunk_dir={}
    chunkindex_dir={}
    i=0
    
    for k in range(0,nf_chunks): 
        chunk_dir["chunk{0}".format(k)]=recognized_text[k][1].split()
        L2 = []
        for j in range(len(chunk_dir["chunk{0}".format(k)])):
            i1=i+j
            L2.append(i1)
        chunkindex_dir["chunk{0}".format(k)]=L2
        i+=chunk_word_limit
    #print(7)
    samp0(lang,chunk_dir,chunkindex_dir,lang_dir,cl_lang_and_index_dir)
    #print(10)
    
    chunk_lang_max=[]
    for i in range(len(chunkindex_dir)):
        L=chunkindex_dir['chunk{0}'.format(i)]
        l1 = []
        if len(L)!=0:
            for i2 in range(len(L)):
                for lang1 in lang:
                    for i3 in range(len(cl_lang_and_index_dir[lang1])):
                        if L[i2]==cl_lang_and_index_dir[lang1][i3][0]:
                            l1.append(lang1)
                    
            most = max(list(map(l1.count,l1)))
            l2=list(set(filter(lambda x: l1.count(x) == most,l1)))
            if len(l2)==1:
                chunk_lang_max.append([i,l2[0]])
            
            else:
                chunk_lang_max.append([i,'None'])
        else:
            chunk_lang_max.append([i,'None'])
    
    
    
    def _prev(chunkindex_dir,current_index,cl_lang_and_index_dir):
        prev_chunk=1
        k=1
        while k!=0:
            if math.floor(current_index/chunk_word_limit) - prev_chunk >= 0:
                if chunkindex_dir['chunk{0}'.format(math.floor(current_index/chunk_word_limit) - prev_chunk)]!=[]:
                    break;
                else:
                    prev_chunk+=1
            else:
                k=0
        if math.floor(current_index/chunk_word_limit)==0:
            prev_chunk = 0
            
        if(math.floor(current_index/chunk_word_limit) - prev_chunk)>=0:
            prev_index = chunkindex_dir['chunk{0}'.format(math.floor(current_index/chunk_word_limit) - prev_chunk)][-1]
            for lang1 in lang:
                for lang2 in cl_lang_and_index_dir[lang1]:
                    if lang2[0]==prev_index:
                        prev_lang1 = lang1
        else:
            prev_lang1='next'
        if prev_lang1=='mixed':
            prev_lang1='next'
        return prev_lang1
    
    def _next(chunkindex_dir,current_index,cl_lang_and_index_dir):
        next_chunk=1
        k=1
        while k!=0:
            if math.floor(current_index/chunk_word_limit) + next_chunk < len(chunkindex_dir):
                if chunkindex_dir['chunk{0}'.format(math.floor(current_index/chunk_word_limit) + next_chunk)]!=[] :
                    break;
                else:
                    next_chunk+=1
            else:
                k=0
               
        #print(next_chunk)
        #print(str(math.floor(current_index/chunk_word_limit))+'error')
        if math.floor(current_index/chunk_word_limit) + next_chunk < len(chunkindex_dir):
            next_index = chunkindex_dir['chunk{0}'.format(math.floor(current_index/chunk_word_limit) + next_chunk)][0]
            for lang1 in lang:
                for lang2 in cl_lang_and_index_dir[lang1]:
                    if lang2[0]==next_index:
                        next_lang1 = lang1
        else:
            next_lang1='prev'
        if next_lang1=='mixed':
            next_lang1='prev'
            
        return next_lang1
    
    def _current_lang(current_index,cl_lang_and_index_dir):
        
        for lang1 in lang:
            for lang2 in cl_lang_and_index_dir[lang1]:
                if lang2[0]==current_index:
                    current_lang1 = lang1
        
        return current_lang1
    
            
    for lang1 in lang:
        l2=cl_lang_and_index_dir[lang1]
        dest_lang_index=[]
        for i2 in l2:
            i2[0]
            i = chunkindex_dir['chunk{0}'.format(math.floor(i2[0]/chunk_word_limit))]
            current_word = chunk_dir['chunk{0}'.format(math.floor(i2[0]/chunk_word_limit))][i2[0]%chunk_word_limit]
            if i2[0] == math.floor(i2[0]/chunk_word_limit)*chunk_word_limit:
                if i2[0] == i[-1]:
                    current_lang = lang1
                    prev_lang = _prev(chunkindex_dir,i2[0],cl_lang_and_index_dir)
                    next_lang = _next(chunkindex_dir,i2[0],cl_lang_and_index_dir)
                    if prev_lang=='next' and current_lang!=next_lang and next_lang!='prev':
                        dest_lang_index.append([i2[0],current_word,current_lang,next_lang])                      
                    elif((prev_lang!=current_lang and next_lang=='prev' and  prev_lang!='next') or (prev_lang!=current_lang and current_lang!=next_lang and prev_lang!='next')):                     
                        dest_lang_index.append([i2[0],current_word,current_lang,prev_lang])

                else:
                    current_lang = lang1
                    prev_lang = _prev(chunkindex_dir,i2[0],cl_lang_and_index_dir)
                    next_lang = _current_lang(i2[0]+1,cl_lang_and_index_dir)
                    if prev_lang=='next' and current_lang!=next_lang and next_lang!='prev':
                        dest_lang_index.append([i2[0],current_word,current_lang,next_lang])
                    elif (prev_lang!=current_lang and next_lang=='prev' and  prev_lang!='next'):
                        dest_lang_index.append([i2[0],current_word,current_lang,prev_lang])
                    elif(prev_lang!=current_lang and current_lang!=next_lang):
                        dest_lang = chunk_lang_max[math.floor(i2[0]/chunk_word_limit)][1]
                        if dest_lang!='None':
                            dest_lang_index.append([i2[0],current_word,current_lang,dest_lang])
                        else:
                            dest_lang_index.append([i2[0],current_word,current_lang,prev_lang])
            
    
            elif(i2[0] == i[-1]):
                current_lang = lang1
                next_lang = _next(chunkindex_dir,i2[0],cl_lang_and_index_dir)
                prev_lang = _current_lang(i2[0]-1,cl_lang_and_index_dir)
                if prev_lang=='next' and current_lang!=next_lang and next_lang!='prev' :
                    dest_lang_index.append([i2[0],current_word,current_lang,next_lang])
                elif(prev_lang!=current_lang and next_lang=='prev' and  prev_lang!='next'):
                    dest_lang_index.append([i2[0],current_word,current_lang,prev_lang])
                elif(prev_lang!=current_lang and current_lang!=next_lang):
                    dest_lang = chunk_lang_max[math.floor(i2[0]/chunk_word_limit)][1]
                    if dest_lang!='None':
                        dest_lang_index.append([i2[0],current_word,current_lang,dest_lang])
                    else:
                        dest_lang_index.append([i2[0],current_word,current_lang,prev_lang])
                    
            else:
                current_lang =lang1
                next_lang = _current_lang(i2[0]+1,cl_lang_and_index_dir)
                prev_lang = _current_lang(i2[0]-1,cl_lang_and_index_dir)
                if(prev_lang!=current_lang and current_lang!=next_lang):
                    dest_lang = chunk_lang_max[math.floor(i2[0]/chunk_word_limit)][1]
                    if dest_lang!='None':
                        dest_lang_index.append([i2[0],current_word,current_lang,dest_lang])
                    else:
                        dest_lang_index.append([i2[0],current_word,current_lang,prev_lang])
                
        #print(cl_lang_and_index_dir)
        for j1 in dest_lang_index:
            L = cl_lang_and_index_dir[j1[2]]
            for j in range(0,len(L)):
                if j1[0] == L[j][0]:
                    L.pop(j)
                    #print('popped')
                    break;
                    
            L = cl_lang_and_index_dir[j1[3]]
            #print('appended')
            L.append(j1[0:2])
            L = sorted(L,key=lambda x:x[0])
            cl_lang_and_index_dir[j1[3]]=L
            
        for lang2 in lang:
            L = cl_lang_and_index_dir[lang2]
            L = [k for k in L if len(k)>0]
            cl_lang_and_index_dir[lang2]=L
        #print(cl_lang_and_index_dir)
        
        
        
    global start_end_list
    start_end_list=[]
    
    def samp1(lang1,chunkindex_dir):
        
        if(len(lang1)!=0):
            if (len(lang1))==1:
                start_end_list.extend((lang1[0][0],lang1[0][0]))
            else:
                if((lang1[0][0]+1)!=lang1[1][0]):
                    start_end_list.extend((lang1[0][0],lang1[0][0],lang1[1][0]))
                else:
                    start_end_list.append(lang1[0][0])
                    
            for m in range(1,len(lang1)-1):
                if((lang1[m][0]+1)!=lang1[m+1][0]):
                    if(lang1[m][0]==chunkindex_dir["chunk{0}".format(math.floor(lang1[m][0]/chunk_word_limit))][-1]):
                        next_chunk=1
                        k=1
                        while k!=0:
                            if chunkindex_dir['chunk{0}'.format(math.floor(lang1[m][0]/chunk_word_limit) + next_chunk)]==[]:
                                if math.floor(lang1[m][0]/chunk_word_limit) != nf_chunks -1:
                                    next_chunk+=1
                            else:
                                k=0
                        if(((math.floor(lang1[m][0]/chunk_word_limit)*chunk_word_limit) + next_chunk*chunk_word_limit)!= lang1[m+1][0]):
                            start_end_list.extend((lang1[m][0],lang1[m+1][0]))
                    else:
                        start_end_list.extend((lang1[m][0],lang1[m+1][0]))
            start_end_list.append(lang1[-1][0])
    
    for i in lang:
        samp1(cl_lang_and_index_dir["{0}".format(i)],chunkindex_dir)
    start_end_list.sort()
    #print(start_end_list)
    i1=0
    i2=chunk_word_limit
    chunklang_startend_dir={}
    for i3 in range(0,nf_chunks):
        m0=[]
        for i4 in range(0,len(start_end_list)):
            if (start_end_list[i4] in range(i1,i2)):
                m0.append(start_end_list[i4])
                
        chunklang_startend_dir["chunk{0}".format(i3)]=m0
        i1+=chunk_word_limit
        i2+=chunk_word_limit
    #print(chunklang_startend_dir)
    return chunk_dir,chunklang_startend_dir,cl_lang_and_index_dir,chunkindex_dir,start_end_list




def lang_output(nf_chunks,chunk_word_limit,lang,chunk_dir,chunk_dur,chunklang_startend_dir,cl_lang_and_index_dir,recognized_text):
    #print(chunk_dir)
    
    lang_out=[]
    dur_out=[]
    sum1=0
    
    for j in range(0,len(chunk_dir)):
        str1=' '.join(str(i) for i in chunk_dir["chunk{0}".format(j)])
        if(j>=1):
            sum1+=chunk_dur[j-1]
        z=str1
        r5=0
        m1=chunklang_startend_dir["chunk{0}".format(j)]
    
        for i in range(0,len(m1)):
            #print(m1)
            #print(m1[0])
            r6=r5
            
            if(i%2==0):
                
                for lang1 in range(0,len(lang)):
                    lang2=[]
                    for lang3 in range(0,len(cl_lang_and_index_dir[lang[lang1]])):
                        lang2.append(cl_lang_and_index_dir[lang[lang1]][lang3][0])
                    if(m1[i] in lang2):
                        lang_out.append("<"+lang[lang1]+">")
                        
                if(len(m1)%2!=0):
                    str2=' '.join(str(ele) for ele in chunk_dir["chunk{0}".format(j+1)])     
                    z3=z[slice(r6,len(z))]
                    #print(z3,"z3 is")
                    #print(chunk_dir["chunk{0}".format(j)][m1[i]%chunk_word_limit])
                    z1=re.search(r'\b({})\b'.format(chunk_dir["chunk{0}".format(j)][m1[i]%chunk_word_limit]),z3)
                    r4=z1.start()+(r6)
     
                    start = float((r4) * chunk_dur[j]/len(str1) )+sum1
                    
                    if(i==len(m1)-1):
                        next_chunk=j+1
                        while(len(chunklang_startend_dir["chunk{0}".format(next_chunk)])==0):
                            next_chunk+=1
                        sum2=0
                        m1=chunklang_startend_dir["chunk{0}".format(next_chunk)]
                        #print(m1)
                        #print(m1[0])
                        #print(j,next_chunk)
                        for b in range(j,next_chunk):
                            sum2+=chunk_dur[b]
                        str2=' '.join(str(ele) for ele in chunk_dir["chunk{0}".format(next_chunk)])
                        z2=re.search(r'\b({})\b'.format(chunk_dir["chunk{0}".format(next_chunk)][m1[0]%chunk_word_limit]),str2)
                        r5=z2.start()+(len(chunk_dir["chunk{0}".format(next_chunk)][m1[0]%chunk_word_limit])-1)+(r6)+len(str1)
                        del chunklang_startend_dir["chunk{0}".format(next_chunk)][0]
                        
                        end = float((r5) * chunk_dur[j]/r5)+sum1+sum2
                        
                        #print(chunk_len_dir["ckl{0}".format(j+1)])
                    else:
                        z2=re.search(r'\b({})\b'.format(chunk_dir["chunk{0}".format(j)][m1[i+1]%chunk_word_limit]),z3)
                        r5=z2.start()+(len(chunk_dir["chunk{0}".format(j)][m1[i+1]%chunk_word_limit])-1)+(r6)+2
                        
                        start = float((r4) * chunk_dur[j]/len(str1) )+sum1
                        
                        end = float((r5) * chunk_dur[j]/len(str1) )+sum1
                        
                            
                else:
                       
                    #print()
                    #print(r6)
                    z3=z[slice(r6,len(z))]
                    #print(z3,"z3 is")
                    #print(chunk_dir["chunk{0}".format(j)][m1[i]%100])
                    #print(m1)
                    #print(chunk_dir["chunk{0}".format(j)])
                    z1=re.search(r'\b({})\b'.format(chunk_dir["chunk{0}".format(j)][m1[i]%chunk_word_limit]),z3)
                    #print(z1.start())
                    r4=z1.start()+(r6)
                    z2=re.search(r'\b({})\b'.format(chunk_dir["chunk{0}".format(j)][m1[i+1]%chunk_word_limit]),z3)
                    r5=z2.start()+(len(chunk_dir["chunk{0}".format(j)][m1[i+1]%chunk_word_limit])-1)+(r6)+2
                    #print(z2)
                    #print(r4,r5)
                    
                    start = float((r4) * chunk_dur[j]/len(str1) )+sum1
                    
                    end = float((r5) * chunk_dur[j]/len(str1) )+sum1
                    
                dur_out.append([start,end])
    print("Hey User your Result Going to ready") 
                
    #print("duration is:",dur)         
                
    return lang_out,dur_out



noise_reduction=True;clapping_detection=False
  
input_audio=input("Enter the audio file name: ")
#print(0)
    #wave_read = wave.open(input_audio,mode='rb')
    
audio_data,rate = sf.read(input_audio)
dur=(len(audio_data)+1)/float(rate)
if(audio_data.ndim > 1):
    audio_data=np.delete(audio_data,slice(1),1)
    audio_data=np.delete(audio_data,slice(1),None)
audio_data=audio_data/32768
#print(1)
chunk_silent=100
if noise_reduction==True:
    noise_reduce(audio_data,rate,win_length=5000,amp_adjust=0.005)
    chunk_rate,nf_chunks,silence_ranges=segmentation(min_silence_len=300,silence_thresh=-50,chunk_silent=chunk_silent)
else:
    audio_data = np.int16(audio_data/np.max(np.abs(audio_data))*32768)
    write('test212.wav', rate,audio_data)
    #print(2)
    chunk_rate,nf_chunks,silent_ranges=segmentation(min_silence_len=500,silence_thresh=-50,chunk_silent=chunk_silent)

recognized_text=[]
recognizing_text(nf_chunks,recognized_text)

lang,lang_dir=lang_files()

recognized_text=sorted(recognized_text,key=lambda x:x[0])


global chunk_word_limit
chunk_word_limit = 1000
chunk_dur=chunk_time(nf_chunks,recognized_text,silent_ranges)

chunk_dir,chunklang_startend_dir,cl_lang_and_index_dir,chunkindex_dir,start_end_list = lang_classify(lang,lang_dir,nf_chunks,chunk_word_limit,recognized_text)

lang_out,dur_out = lang_output(nf_chunks,chunk_word_limit,lang,chunk_dir,chunk_dur,chunklang_startend_dir,cl_lang_and_index_dir,recognized_text)
lang_out1,dur_out1=lang_correction(lang_out,dur_out)
jkl=open("result.txt","w")
jkl.close()
jkl=open("result.txt","a")
jkl.write("Input Audio:"+str(input_audio))
jkl.write("\nFormat is:\nDuration\nLanguage\n")
print("Format is \nDuration\nLanguage")
jkl.write("The duration of the given input audio file is:"+" "+str(dur)+"sec\n")
print("The duration of the given input audio file is:",dur)
time.sleep(2)
print("Your Result is Ready:")
time.sleep(3)
for gn in range(len(lang_out1)):
    jkl.write(str(dur_out1[gn][0])+"\n")
    jkl.write(str(dur_out1[gn][1])+"\n")
    jkl.write(str(lang_out1[gn])+"\n")
    print(dur_out1[gn][0])
    print(dur_out1[gn][1])
    print(lang_out1[gn])
jkl.close()





















    



























