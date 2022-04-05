import cv2
import numpy as np

#front = cv2.FONT_HERSHEY_SIMPLEX
front = cv2.FONT_HERSHEY_TRIPLEX        
# front = cv2.FONT_HERSHEY_PLAIN

# Happiness - AU 6, 12 = df[4], df[9]
# Sadness - AU 1,4,15 = df[0], df[2], df[11]
# Surprise - AU 1,2,5,26 = df[0], df[1], df[3], df[17]
# Fear - AU 1,2,4,5,7,20,26 = df[0], df[1], df[2], df[3], df[5], df[13], df[17]
# Anger - AU 4,5,7,23 = df[2], df[3], df[5], df[14]
# Disgust - AU 9,15,16 = df[6], df[11]
# Contempt - AU 12,14 = df[9], df[10]

def plotau(au_svm, au_logistic, au_rf, au_JAANET, au_DRML, auoccur_col):
    # img = np.zeros((540, 540, 3), np.uint8)
    img = np.zeros((540+50, 740+500, 3), np.uint8)
    img.fill(255)

    listname = ['AU01 - Inner brow raiser ', #0
                'AU02 - Outer brow raiser ', #1
                'AU04 - Brow lowerer ', #2
                'AU05 - Upper lid raiser ', #3
                'AU06 - Cheek raiser ', #4
                'AU07 - Lid tightener ', #5
                'AU09 - Nose wrinkler ', #6
                'AU10 - Upper lip raiser ', #7
                'AU11 - Nasolabial deepener ', #8
                'AU12 - Lip corner puller ', #9
                'AU14 - Dimpler ', #10
                'AU15 - Lip corner depressor ', #11
                'AU17 - Chin raiser ', #12
                'AU20 - Lip stretcher ', #13
                'AU23 - Lip tightener ', #14
                'AU24 - Lip pressor ',#15
                'AU25 - Lips part ', #16
                'AU26 - Jaw drop ', #17
                'AU28 - Lip suck ', #18
                'AU43 - Eyes closed '] #19
                
    jaa_drml_au = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']                

    listemo = [ "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral",]

    ####################### Action units SVM'
    SVM_dict = {}
    cv2.putText(img, 'SVM Classification', (10, 15), front, 0.4, (0, 0, 51), 1)
    cv2.rectangle(img, (10, 20), (232+60, 514), (224, 224, 224), 1)
    for i in range(0,20):
        cv2.putText(img, str(listname[i]) , (20, 24*(i+1)+10), front, 0.35, (0, 0, 51), 1)
        cv2.rectangle(img, (160+50, 24*(i+1)), (230+50, 24*(i+1)+10), (255, 255, 204), cv2.FILLED) #Frame
        cv2.rectangle(img, (160+50, 24*(i+1)), (160+50 + int(au_svm[i] * 70), 24*(i+1)+10), (153, 153, 0), cv2.FILLED) #energy
        SVM_dict[listname[i]] = round(au_svm[i],3)
    #print(SVM_dict)

    ####################### Action units : logistic Regression
    logistic_dict = {}
    cv2.putText(img, 'logistic Regression', (240+50, 15), front, 0.4, (0, 0, 51), 1)
    cv2.rectangle(img, (240+50, 20), (510+60, 514), (224, 224, 224), 1)
    for i in range(0,20):
        cv2.putText(img, str(listname[i]) , (250+50, 24*(i+1)+10), front, 0.35, (0, 0, 51), 1) 
        cv2.rectangle(img, (430+50, 24*(i+1)), (500+50, 24*(i+1)+10), (255, 255, 204), cv2.FILLED) #Frame
        cv2.rectangle(img, (430+50, 24*(i+1)), (430+50 + int(au_logistic[i] * 70), 24*(i+1)+10), (153, 153, 0), cv2.FILLED) #energy
        logistic_dict[listname[i]] = round(au_logistic[i],3)
    #print(logistic_dict)

    ####################### Action units : rf Regression
    rf_dict = {}
    cv2.putText(img, 'rf Regression', (520+60, 15), front, 0.4, (0, 0, 51), 1)
    cv2.rectangle(img, (520+60, 20), (770+65, 514), (224, 224, 224), 1)
    for i in range(0,20):
        cv2.putText(img, str(listname[i]) , (520+60, 24*(i+1)+10), front, 0.35, (0, 0, 51), 1)
        cv2.rectangle(img, (700+50, 24*(i+1)), (770+50, 24*(i+1)+10), (255, 255, 204), cv2.FILLED) #Frame
        cv2.rectangle(img, (700+50, 24*(i+1)), (700+50 + int(au_rf[i] * 70), 24*(i+1)+10), (153, 153, 0), cv2.FILLED) #energy
        rf_dict[listname[i]] = round(au_rf[i],3)
    #print(rf_dict)
    
        
    ####################### Action units : JAANET
    JAANET_dict = {}
    cv2.putText(img, 'JAANET', (700+140, 15), front, 0.4, (0, 0, 51), 1)
    cv2.rectangle(img, (700+140, 20), (760+210, 514), (224, 224, 224), 1)
    for i in range(0,12):
        cv2.putText(img, str(jaa_drml_au[i]) , (700+150, 24*(i+1)+10), front, 0.35, (0, 0, 51), 1)
        cv2.rectangle(img, (690+200, 24*(i+1)), (760+200, 24*(i+1)+10), (255, 255, 204), cv2.FILLED) #Frame
        cv2.rectangle(img, (690+200, 24*(i+1)), (690+200 + int(au_JAANET[i] * 70), 24*(i+1)+10), (153, 153, 0), cv2.FILLED) #energy
        JAANET_dict[jaa_drml_au[i]] = round(au_JAANET[i],3)
    #print(JAANET_dict)
        
    ####################### Action units : DRML
    DRML_dict1 = {}
    DRML_dict2 = {}
    cv2.putText(img, 'DRML', (720+260, 15), front, 0.4, (0, 0, 51), 1)
    cv2.rectangle(img, (720+255, 20), (780+315, 514), (224, 224, 224), 1)
    for i in range(0,12):
        cv2.putText(img, str(jaa_drml_au[i]) , (720+260, 24*(i+1)+10), front, 0.35, (0, 0, 51), 1)
        cv2.rectangle(img, (710+310, 24*(i+1)), (780+310, 24*(i+1)+10), (255, 255, 204), cv2.FILLED) #Frame
        cv2.rectangle(img, (710+310, 24*(i+1)), (710+310 + int(abs(au_DRML[0][i]) * 10), 24*(i+1)+10), (153, 153, 0), cv2.FILLED) #energy
        DRML_dict1[jaa_drml_au[i]] = round(au_DRML[0][i],3)
        DRML_dict2[jaa_drml_au[i]] = round(au_DRML[1][i],3)
    #print(DRML_dict)    

    result = {'svm': SVM_dict, 'logistic':logistic_dict, 'rf':rf_dict, 'jaanet':JAANET_dict,  'DRML':[DRML_dict1, DRML_dict2]}
    #print(result)
    return img, result
    
    
    


