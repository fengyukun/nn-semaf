import sys,os,re,math

def get_common_verbs(eval_repo,path,ext):
  files=os.listdir(path)
  run_dict = []
  for f in files:
    if f.endswith(ext):
      run_dict.append(f.split(ext)[0])
  if len(run_dict)==0:
    sys.stderr.write('No file with extension '+ext+' found\n')
  files=os.listdir(eval_repo)
  eval_dict=[]
  for f in files:
    if f.endswith(ext) and f.split(ext)[0] in run_dict:
      eval_dict.append(f.split(ext)[0])
  if len(eval_dict)==0:
    sys.stderr.write('No verb matched with gold standard\n')
  return eval_dict,len(files)

def eval_task(gold_dir, path,task,detail=False):
  #Reads all files with relevant ext from run repo and returns run score
  task_gold_repo = {'1':gold_dir+'/task1','2':gold_dir+'/task2','3':gold_dir+'/task3'}
  task_ext = {'1':'.parse','2':'.clust','3':'.pat'}
  eval_repo = task_gold_repo[task]
  ext=task_ext[task]
  eval_dict,n_v = get_common_verbs(eval_repo,path,ext)
  sys.stderr.write(str(len(eval_dict))+' verb files found in '+path+'\n')

  glob = {}
  detail_eval = {}
  n_s = 0
  if task == '1':
    #initialize
    for i in ['syn','sem']:
      glob[i]={}
    for i in ['ref','run','cor']:
      glob['syn'][i]=0
      glob['sem'][i]=0
    for f in eval_dict:
      run = store_task1_data(path+'/'+f+ext)
      gold = store_task1_data(eval_repo+'/'+f+ext)
      verb = get_score_values_task1(run,gold)
      score = metric_task1(f,verb,detail)
      n_s +=score
      if detail:
        print 'score['+f+']=',score
  elif task == '2':
    for f in eval_dict:
      run = store_task2_data(path+'/'+f+ext)
      gold = store_task2_data(eval_repo+'/'+f+ext)
      score = bcubed(f,run,gold,detail)
      if detail:
        print 'score['+f+']=',score
      n_s+=score
  elif task == '3':
    for f in eval_dict:
      run = store_task3_data(path+'/'+f+ext)
      gold = store_task3_data(eval_repo+'/'+f+ext)
      threshold = round(len(gold)*1.5)
      if len(run)>threshold:
        run = run[0:threshold]
      score = get_score_values_task3(run,gold)
      if detail:
        print 'score['+f+']=',score
      n_s+=score
  print 'Task '+task+' AVERAGE score:',n_s/n_v

def get_score_values_task3(run,gold):
  score = {}
  id2pat = {}
  onto = get_ontology()
  for i in run:
    for j in gold:
      cor = subst = ins = deleted = 0
      for k in run[i]:
        if run[i][k] != '' and k not in ['verb','pat']:
          if gold[j][k] != '':
            if run[i][k] == gold[j][k]:
              cor +=1
            elif gold[j][k] in onto and run[i][k] in onto and onto[run[i][k]].get(gold[j][k],'none') !='none':
              cor+=0.25
              subst+=0.75
            elif gold[j][k] in onto and run[i][k] in onto and onto[gold[j][k]].get(run[i][k],'none') != 'none' and run[i][k]!= 'Anything' and onto[run[i][k]]!= 'Anything':
              cor+=0.5
              subst+=0.5
            else:
              subst +=1
          else:
            ins +=1
      for k in gold[j]:
        if gold[j][k] != '' and run[i][k] == '' and k not in ['verb','pat']:
          deleted +=1
        if k == 'pat':
          id2pat[j]=gold[j][k]
      prec = rec = f1 = 0
      if (cor+subst+ins) > 0 and cor>0:
        prec = cor / float(cor+subst+ins)
      if (cor+subst+deleted) > 0 and cor>0:
        rec = cor / float(cor+subst+deleted)
      if prec>0 or rec>0:
        f1 = 2*prec*rec / float(prec + rec)
      if score.get(j,'none') == 'none':
        score[j] = {}
      score[j][i] = f1
  selected = {}
  final = {}
  for pat_gold in sorted(score):
    found = 0
    for pat_run in sorted(score[pat_gold], key=score[pat_gold].get,reverse=True):
      if selected.get(pat_run,'none') == 'none' and found == 0:
        selected[pat_run] = 1
        final[pat_gold] = score[pat_gold][pat_run]
        found = 1
  fscore = ftot = 0
  for j in gold:
    if final.get(j,'none')!='none':
      fscore += final[j]
    ftot += 1
  verb_score=fscore/float(ftot)
  return verb_score

def get_ontology():
  import json
  data = json.loads(open("CPA-ontology.json").read())
  onto = {}
  def build_onto(node,parent,onto,n):
    if 'term' in node:
      st = node['term']
      if n == 1:
        onto[n] = {'parent': parent, 'name':st, 'id':n}
      else:
        onto[n] = {'parent': parent, 'name':st, 'id':n}
    if 'sub' in node:
      p = n
      for j in range(0,len(node['sub'])):
        n = build_onto(node['sub'][j],p,onto,n+1)
    return n
  n = build_onto(data[0],0,onto,1)
  ontoterms = {}
  for c in onto:
   term = onto[c]['name']
   if ontoterms.get(term,'none') == 'none':
     ontoterms[term] = {}
   for d in onto:
     term2 = onto[d]['name']
     if d != c:
       if onto[c]['parent'] == d:
         ontoterms[term][term2] = 1
  return onto

def bcubed(verb,run,gold,detail):
  prec = metric_task2_alt(run,gold,detail)
  rec = metric_task2_alt(gold,run,detail)
  f0 = 0
  if (prec+rec)>0:
    f0 = (2*prec*rec)/(prec+rec)
  if detail:
    print verb, prec, rec, f0
  return f0

def metric_task2(run,gold,detail):
  F=P=Q={}
  norm = 1/(math.sqrt(len(run)*len(gold))-1)
  sumconc = 0
  for I in run:
    F[I]={}
    P=len(run[I])
    for J in gold:
      F[I][J]=0
      Q=len(gold[J])
      for i in run[I]:
        if i in gold[J]:
          F[I][J]+=1 #f_ij
      conc = (F[I][J]*F[I][J])/float(Q*P)
      #if detail:
        #sys.stderr.write(str(I)+' '+str(J)+'='+str(conc)+' // ')
      sumconc += conc
  moc = (sumconc-1)*norm
  return moc

def metric_task2_alt(run, gold, detail):
  Prec = {}
  #build ingold
  ingold = inversedict(gold)
  overall = 0
  overall_prec_inst = 0
  overall_prec_class = 0
  for cl in run:
    overall +=len(run[cl])
    cor_class = 0
    totalcl=0
    for inst in run[cl]:
      cor_inst = 0
      for inst2 in run[cl]:
        totalcl+=1
        if inst in ingold and inst2 in ingold and ingold[inst]==ingold[inst2] :#and ingold[inst] == cl:
          cor_inst += 1
          cor_class += 1
      prec_inst = cor_inst / float(len(run[cl]))
      overall_prec_inst+=prec_inst
      Prec[inst] = prec_inst
    prec_class = cor_class / float(totalcl)
    overall_prec_class+=prec_class
  if overall>0:
    ave_P_inst = overall_prec_inst / float(overall)
    ave_P_class = overall_prec_class / float(len(run))
    return ave_P_inst
    #return ave_P_class
  return 0

def inversedict(gold):
  ingold = {}
  for i in gold:
    for n in gold[i]:
      ingold[n]=i
  return ingold

def metric_task1(f,dat, detail=False):
  glob_f1 = 0
  for val in dat:
   allrec=allprec=allf1=allcor=allrun=allref=0
   n=0
   for i in dat[val]:
    n+=1
    data = dat[val][i]
    prec = 0
    rec = 0
    f1 = 0
    if data.get('ref','none')=='none':
      continue
    if data.get('cor','none')=='none':
      data['cor']=0
    if data.get('run','none')=='none':
      data['run']=0
    if data['run']>0:
      rec = data['cor'] / float(data['run'])
    prec = data['cor'] / float(data['ref'])
    if (prec+rec) > 0:
      f1 = (2 * prec * rec) / (prec + rec)
    allf1 += f1
    allrec+=rec
    allprec+=prec
    allcor+=data['cor']
    allref+=data['ref']
    allrun+=data['run']
   if detail:
    print f,val,allcor,allref,allrun,allprec/float(n),allrec/float(n),allf1/float(n)
   glob_f1+=allf1/float(n)
  return glob_f1/2

def get_score_values_task1(run,gold):
  #compute score for synt and sem separately and then merge
  detail_dict = {'syn':{},'sem':{}}
  for val in ['syn','sem']:
    for tok in run[val]:
      if gold[val].get(tok,'none')=='none':
        if detail_dict[val].get(run[val][tok],'none') =='none':
            detail_dict[val][run[val][tok]]={}
        try:
            detail_dict[val][run[val][tok]]['run']+=1
        except KeyError:
            detail_dict[val][run[val][tok]]['run']=1
    for tok in gold[val]:
      if detail_dict[val].get(gold[val][tok],'none') =='none':
          detail_dict[val][gold[val][tok]]={}
      try:
          detail_dict[val][gold[val][tok]]['ref']+=1
      except KeyError:
          detail_dict[val][gold[val][tok]]['ref']=1
      if run[val].get(tok,'none')!='none':
        try:
            detail_dict[val][gold[val][tok]]['run']+=1
        except KeyError:
            detail_dict[val][gold[val][tok]]['run']=1
        if run[val][tok]==gold[val][tok]:
          try:
              detail_dict[val][gold[val][tok]]['cor']+=1
          except KeyError:
              detail_dict[val][gold[val][tok]]['cor']=1
  return detail_dict

def store_task1_data(parse_file):
  data = {}
  data['syn'] = {}
  data['sem'] = {}
  with open(parse_file) as f:
    lines = f.readlines()
    for l in lines:
      if l.rstrip() =='':
        continue
      if len(l.split('\t'))!=4:
        sys.stderr.write('Format issue with file '+parse_file+' at:\n'+l)
      fields = l.split('\n')[0].split('\t')
      if fields[2]=='v':
        continue
      if fields[2]!='':
        data['syn'][fields[0]]=fields[2]
      if fields[3]!='':
        data['sem'][fields[0]]=fields[3]
  return data

def store_task2_data(clust_file):
  data = {}
  with open(clust_file) as f:
    lines = f.readlines()
    for l in lines:
      if l.rstrip() =='':
        continue
      if len(l.split('\t'))!=2:
        sys.exit('Format issue with file '+clust_file)
      fields = l.split('\n')[0].split('\t')
      try:
        b = int(fields[1])
        if fields[1]!='' and isinstance(b,int):
          try:
            data[fields[1]].append(fields[0])
          except KeyError:
            data[fields[1]]=[]
            data[fields[1]].append(fields[0])
      except:
        continue
  return data

def store_task3_data(pat_file):
  data = {}
  with open(pat_file) as f:
    lines = f.readlines()
    n_pat = 0
    for l in lines:
      if l.rstrip() =='':
        continue
      if len(l.split('\t'))!=9:#changed to 9 by subtracting verb and pattern
        sys.exit('Format issue with file '+pat_file+', length of line = '+str(len(l.split('\t'))))
      fields = l.split('\n')[0].split('\t')
      n_pat += 1
      data[n_pat] = {'subject':fields[0], 'object':fields[1],'indirect_object':fields[2],
                     'noun_adjective_complement':fields[3],'verb_complement':fields[4],'preposition_1':fields[5],
                     'adverbial_complement_1':fields[6],'preposition_2':fields[7],'adverbial_complement_2':fields[8]}
  return data

def main(argv):
  if len(argv) != 4:
    sys.exit('\n     ### SEMEVAL 2015 TASK 15 Scorer ###\n\n Please provide 3 arguments:\n  1: the number of the subtask you wish to evaluate,\n  2: the directory where the run files are stored,\n  3: the directory where the gold standard file directories like task1, task2, and task3 are stored.\n\n For example, to test your run files of task1, use \n\n       python scorer.py 1 ../run ../task1\n')
  if argv[1] not in ['1','2','3']:
    sys.exit('\n     ### SEMEVAL 2015 TASK 15 Scorer ###\n\n Please provide 3 arguments:\n  1: the number of the subtask you wish to evaluate,\n  2: the directory where the run files are stored,\n  3: the directory where the gold standard file directories like task1, task2, and task3 are stored.\n\n For example, to test your run files of task1, use \n\n       python scorer.py 1 ../run ../task1\n')
  detail = False
  eval_task(argv[3],argv[2],argv[1],detail)

if __name__ == "__main__":
    sys.exit(main(sys.argv))

