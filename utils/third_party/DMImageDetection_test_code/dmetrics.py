'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    t = np.argmin(np.abs(1.-tpr-fpr))
    
    return (fpr[t] + 1 - tpr[t])/2


def pd_at_far(y_true, y_score, fpr_th):
    '''
    Returns the Pd at fixed FAR for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.interp(fpr_th, fpr, tpr)


def macc(y_true, y_score):
    '''
    Returns the maximum accuracy for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.max(tpr+1-fpr)/2
