import argparse
from socket import gethostname

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    # This handler does retries when HTTP status 429 is returned
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler
except ImportError:
    import subprocess
    import sys
    
    subprocess.call([sys.executable, "-m", "pip", "install", "--user",
                     "pip install slack_sdk"])
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    # This handler does retries when HTTP status 429 is returned
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

try:
    import psutil
except ImportError:
    import subprocess
    import sys
    
    subprocess.call([sys.executable, "-m", "pip", "install", "--user",
                     "psutil"])
    import psutil


def _print_error(e: SlackApiError):
    # You will get a SlackApiError if "ok" is False
    assert e.response["ok"] is False
    assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
    print(f"Got an error: {e.response['error']}")


class ISPLSlack:
    """
    Class to post a custom message to polimi-ispl workspace on Slack
    """
    
    def __init__(self):
        # polimi-ispl workspace
        self.client = WebClient(token='xoxp-335564641463-335765559414-423004379767'
                                      '-3314e02838c0aa641ac5f7648cff34eb')
        # enable rate limited error retries as well
        self.client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=1))

        self.machine = gethostname()
        self.icon = ':desktop_computer:'
        self.channels = [channel['name'] for channel in
                         self.client.conversations_list()['channels']]
        self.users = {user['name']: user['id'] for user in
                      self.client.users_list()['members']}
    
    def to_channel(self, channel, message, tags=None, file=None):
        if tags is not None:
            message = ', '.join(['<@' + self.users[t] + '>' for t in tags]) \
                      + '\n' + message
        
        if channel in self.channels:
            if file is not None:
                try:
                    response = self.client.files_upload(channels=channel,
                                                        initial_comment=message,
                                                        username=self.machine,
                                                        icon_emoji=self.icon,
                                                        file=file)
                    assert response["file"]
                except SlackApiError as e:
                    _print_error(e)
            else:
                try:
                    response = self.client.chat_postMessage(channel=channel,
                                                            text=message,
                                                            username=self.machine,
                                                            icon_emoji=self.icon)
                    assert response["message"]["text"] == message
                except SlackApiError as e:
                    _print_error(e)
        else:
            raise KeyError('Channel ' + channel + ' does not exist!')
    
    def to_user(self, recipient, message, file=None):
        
        if message is None:
            raise ValueError('The message cannot be empty!')
        
        if recipient in self.users.keys():
            if file is not None:
                try:
                    response = self.client.files_upload(channels=self.users[recipient],
                                                        initial_comment=message,
                                                        username=self.machine,
                                                        icon_emoji=self.icon,
                                                        user=self.users['slackbot'],
                                                        file=file)
                    assert response["file"]
                except SlackApiError as e:
                    _print_error(e)
            else:
                try:
                    response = self.client.chat_postMessage(channel=self.users[recipient],
                                                            text=message,
                                                            username=self.machine,
                                                            icon_emoji=self.icon)
                    assert response["message"]["text"] == message
                except SlackApiError as e:
                    _print_error(e)
        else:
            raise KeyError('User ' + recipient + ' does not exist!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', nargs='+', type=str, required=False,
                        help='usernames to be noticed')
    parser.add_argument('-c', nargs='+', type=str, required=False,
                        help='Channels name (without #). Must be public!')
    parser.add_argument('-m', type=str, required=False,
                        help='Message to be sent [default: "I have finished '
                             'PARENT_PROCESS"')
    parser.add_argument('-t', type=int, required=False,
                        help='Elapsed time in seconds (hint: use $SECONDS)')
    parser.add_argument('-f', type=str, required=False,
                        help='Path to file to be attached')
    parser.add_argument('--recipients', action='store_true', default=False,
                        help='Print all the available channels and users')
    
    args = parser.parse_args()
    
    users = args.u
    channels = args.c
    time = args.t
    file = args.f
    message = args.m if args.m is not None else \
        'I have finished `' + psutil.Process().parent().cmdline()[1] + '`'
    
    if time is not None:
        s = time % 60
        m = (time // 60) % 60
        h = time // 3600
        timestamp = '%dh:%dm:%ds' % (h, m, s)
        linking_msg = ' in ' if args.m is None else ' '
        message += linking_msg + timestamp
    
    slack = ISPLSlack()
    
    if args.recipients:
        print("Users:\t", ", ".join(slack.users))
        print("Channels:\t", ", ".join(slack.channels))
    
    if channels is not None:  # pst message to channels tagging the users
        for c in channels:
            try:
                slack.to_channel(channel=c, message=message, tags=users, file=file)
            except KeyError:
                pass
    else:  # send directly to the users
        if users is not None:
            for user in users:
                try:
                    slack.to_user(recipient=user, message=message, file=file)
                except KeyError:
                    pass
    return True


if __name__ == '__main__':
    main()