import os
import sys

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)

def main(args=sys.argv[1:]):
    # Delete previous iterations
    session_dir = str(args[0])
    if not os.path.exists(session_dir):
        print(f"model not found! {session_dir}")
        return
    stage = str(args[1])
    end = int(args[2])
    exclude = list(map(int, args[3:]))
    for eps in range(1, end):
        if (not eps in exclude):
            delete_file(os.path.join(session_dir, 'actor' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_actor' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'critic' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_critic' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'stage'+ stage +'_episode'+str(eps)+'.json'))
            delete_file(os.path.join(session_dir, 'stage'+ stage +'_episode'+str(eps)+'.pkl'))


if __name__ == '__main__':
    main()