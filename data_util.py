
def readDic(filename, out="str"):
    output_str_idx=dict()
    output_idx_str=dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k,idx=line.strip().split('\t')
            except:
                print(line)
                k,idx=line.strip().split()
            output_str_idx[k]=idx
            output_idx_str[idx]=k
        output_str_idx[len(output_str_idx)] = '<PAD>'
        output_idx_str[len(output_idx_str)] = '<PAD>'
    if out=='str':
        return output_str_idx
    elif out=='idx':
        return output_idx_str
    else:
        return output_str_idx, output_idx_str

def v2conv(conv, istopic=True, isgoal=True, iskg=True):
    """
    Args:
        conv: v2lines[i]
        istopic: text Topic 출력여부
        isgoal: text type(goal)출력여부
        iskg: text kg 출력여부
    Returns: {'uttrs':uttrs, 'roles':roles, 'topics':topics, 'goals':goals, 'kgs':kgs, 'situation':situation, 'user_profile':usr_profile}
    """
    usr_profile=conv.get('user_profile')
    situation=conv.get('situation')
    topics = conv.get('goal_topic_list') if istopic else ["" for _ in range(len(conv.get('goal_type_list')))]
    goals = conv.get('goal_type_list') if isgoal else ["" for _ in range(len(conv.get('goal_type_list')))]
    kgs = conv.get('knowledge')# if iskg else ["" for _ in range(len(conv.get('goal_type_list')))]
    uttrs = [i if i[0] != '[' else i[4:] for i in conv.get('conversation') ] # utterance 내 [1] 과 같은 형태 삭제
    roles=["system",'user'] if goals[0]=='Greetings' else ['user','system']
    for i in range(len(kgs)-2): roles.append(roles[i%2])
    return {'uttrs':uttrs, 'roles':roles, 'topics':topics, 'goals':goals, 'kgs':kgs, 'situation':situation, 'user_profile':usr_profile}