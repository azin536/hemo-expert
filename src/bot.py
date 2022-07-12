from discord import Webhook, RequestsWebhookAdapter


WEBHOOK = 'https://discord.com/api/webhooks/993421397521072189/gw186mzZz7wCBDKRcPDvSgd3Ahym8i2C-CywoyxQm5BQ_sv6kU4IlZQNLqtnhH89OBPd'


class DiscordBot:
    """
    this bot is for sending a message to discord channel which consists of experiment name(repository name), run id,
    the epoch that it sends message at the end of and an optional message which consists of the logs and is optional.

    How to :
    bot = DiscordBot("Webhook URL")
    bot.send_message(experiment name, run id, epoch, message)
    """
    def __init__(self, webhook_url: str = WEBHOOK):
        self.url = webhook_url

    def send_message(self, exp_name: str, run_id: int, epoch: int, message=None):
        """
        this method sends request to the discord channel which webhook url is given as input.
        Args:
            - exp_name(str): name of the experiment which is name of the repository + branch
            - run_id(int): run id which is given from active run info.
            - epoch(int): this message will be sent after each epoch, so the number of the epoch is needed.
            - message(dict): this is an optional message which can be the logs.
        """
        webhook = Webhook.from_url(self.url, adapter=RequestsWebhookAdapter())
        content_ = '============================' + '\n' + '**experiment_name: **' + exp_name + '\n' + '**Run_ID: **' \
                   + str(run_id) + '\n' + '**Epoch: **' + str(epoch)
        if message is not None:
            content = content_ + '\n' + '**messages: **' + str(message) + '\n' + '============================'
        else:
            content = content_ + '\n' + '============================'

        webhook.send(content=content)
