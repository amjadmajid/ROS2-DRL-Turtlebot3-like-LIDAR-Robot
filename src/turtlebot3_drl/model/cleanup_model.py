import os
import sys

def delete_file(path):
    print(path)
    if os.path.exists(path):
        os.remove(path)

def main(args=sys.argv[1:]):
    # Delete previous iterations
    session_dir = str(args[0])
    stage = str(4)
    interval = 100
    end = int(args[1])
    for i in range(interval, end, interval):
        if (i % 1000 != 0):
            delete_file(os.path.join(session_dir, 'actor' + '_stage'+str(stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_actor' + '_stage'+str(stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'critic' + '_stage'+str(stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_critic' + '_stage'+str(stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'stage'+str(stage)+'_episode'+str(i)+'.json'))
            delete_file(os.path.join(session_dir, 'stage'+str(stage)+'_episode'+str(i)+'.pkl'))


if __name__ == '__main__':
    main()
