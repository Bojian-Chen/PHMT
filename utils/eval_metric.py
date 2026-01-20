import torch
import pandas as pd

def eval_metric(args,Correct):

    BWT = []
    AA = []
    AF = []
    AG = []
    AP = []
    Correct_t = torch.tensor(Correct)

    for i in range(Correct_t.shape[0]):
        AP.append(Correct_t[i,0:i+1])
        AA.append(Correct_t[i][i])
        BWT.append(Correct_t[-1][i] - Correct[i][i])

    for j in range(Correct_t.shape[0]-1): 
        AF.append( Correct_t[j+1:Correct_t.shape[0],j])
        AG.append( Correct_t[j,j+1:Correct_t.shape[0]])

    
    AP = torch.mean(torch.cat(AP)).item()
    AF = torch.mean(torch.cat(AF)).item()
    AMF = (AF - torch.mean(torch.tensor(AA[:-1]))).item()
    AG = torch.mean(torch.cat(AG)).item()
    AA = torch.mean(torch.tensor(AA)).item()
    BWT = torch.mean(torch.tensor(BWT)).item()
    ACC = torch.mean(Correct_t[-1]).item()
    
    print('AP: {:.2f}%'.format(AP))
    print('AF: {:.2f}%'.format(AF))
    print('AMF: {:.2f}%'.format(AMF))
    print('AG: {:.2f}%'.format(AG))
    print('AA: {:.2f}%'.format(AA))
    print('BWT: {:.2f}%'.format(BWT))
    print('ACC: {:.2f}%'.format(ACC))
    
    if args.save_model:
        results = [
            {'Metric': 'AP', 'Value': AP},
            {'Metric': 'AF', 'Value': AF},
            {'Metric': 'AMF', 'Value': AMF},
            {'Metric': 'AG', 'Value': AG},
            {'Metric': 'AA', 'Value': AA},
            {'Metric': 'BWT', 'Value': BWT},
            {'Metric': 'ACC', 'Value': ACC}
        ]
        df = pd.DataFrame(results)
        df1 = pd.DataFrame(Correct)
        df = pd.concat([df, df1], axis=1)
        df.to_csv(args.pth + 
                  '/' + args.incremental_mode +'_Result.csv', index=False)
    return AP, AF, AMF, AG, AA, BWT, ACC 
