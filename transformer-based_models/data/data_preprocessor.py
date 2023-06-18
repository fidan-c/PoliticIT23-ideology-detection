import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import emoji
import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

TokenizerType = Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]
PREPROCESSOR = TextPreProcessor(
    # entities that will be normalized e.g. @user -> <user>
    normalize=["url", "email", "user"],
    unpack_hashtags=True,
    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    dicts=[emoticons],
)


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, tok: TokenizerType, lang: str) -> None:
        self._tok = tok
        self._lang = lang  # es, it, ...
        self._df = df

        if {"gender", "ideology_binary", "ideology_multiclass"}.issubset(
            self._df.columns
        ):
            self._enc_gender = LabelEncoder().fit(self._df["gender"])
            self._enc_i_bin = LabelEncoder().fit(self._df["ideology_binary"])
            self._enc_i_mul = LabelEncoder().fit(self._df["ideology_multiclass"])

    def _preprocess(self, sample: str):
        sample = emoji.demojize(sample, language=self._lang)
        sample = " ".join(PREPROCESSOR.pre_process_doc(sample))

        patterns = {
            r"<user>|<email>|<url>|<sad>|<seallips>|<happy>": " ",
            r" *[\[\]\"”„“…»«—#$%&’()*+,-<>./:;=@_\\^`{|}~?!°'‘・] *": " ",
            r"POLITICIAN|POLITICAL PARTY|HASHTAG": "",
            r" +": " ",
        }

        for pattern, replacement in patterns.items():
            sample = re.sub(pattern, replacement, sample)

        sample = sample.strip()

        return sample

    def _prepare_sample(
        self,
        label: str,
        joined_tweets: str,
        num_tokens: int,
        user_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if {"gender", "ideology_binary", "ideology_multiclass"}.issubset(
            self._df.columns
        ) and user_data is not None:
            sample = {
                "author": label,
                "num_tokens": num_tokens,
                "gender": user_data["gender"].values[0],
                "ideology_binary": user_data["ideology_binary"].values[0],
                "ideology_multiclass": user_data["ideology_multiclass"].values[0],
                "tweet": joined_tweets,
            }
        else:
            sample = {"author": label, "num_tokens": num_tokens, "tweet": joined_tweets}

        return sample

    def encode_labels(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = dataset.copy()
        df["gender"] = self._enc_gender.transform(df["gender"])
        df["ideology_binary"] = self._enc_i_bin.transform(df["ideology_binary"])
        df["ideology_multiclass"] = self._enc_i_mul.transform(df["ideology_multiclass"])

        return df

    def decode_labels(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = dataset.copy()
        df["gender"] = self._enc_gender.inverse_transform(df["gender"])
        df["ideology_binary"] = self._enc_i_bin.inverse_transform(df["ideology_binary"])
        df["ideology_multiclass"] = self._enc_i_mul.inverse_transform(
            df["ideology_multiclass"]
        )

        return df

    def prepare_data(self, max_tokens: int = 510) -> pd.DataFrame:
        samples = []
        if {"gender", "ideology_binary", "ideology_multiclass"}.issubset(
            self._df.columns
        ):
            df = self.encode_labels(self._df)
        else:
            df = self._df

        df["tweet"] = df["tweet"].apply(self._preprocess)

        # collects all users' IDs (labels)
        user_labels = list(df["label"].unique())

        for label in (pbar := tqdm(user_labels)):
            pbar.set_description(f"Preprocessing data")

            # retrieve all tweets for current user (label)
            user_data = df[df["label"] == label]
            user_tweets = user_data["tweet"].values
            random.shuffle(user_tweets)

            num_tokens = 0
            collected = []

            # group the user tweets into blocks of max_tokens
            for tweet in user_tweets:
                len_tokens_tweet = len(self._tok.tokenize(tweet))

                if len_tokens_tweet > max_tokens:
                    continue

                if (num_tokens + len_tokens_tweet) > max_tokens:
                    # creates a sample from all previously collected tweets;
                    # the current tweet will be included in the next block
                    joined_tweets = " ".join(collected)
                    samples.append(
                        self._prepare_sample(
                            label, joined_tweets, num_tokens, user_data
                        )
                    )
                    num_tokens = len_tokens_tweet
                    # store current tweet for the next block
                    collected = [tweet]
                else:
                    num_tokens += len_tokens_tweet
                    collected.append(tweet)

            # if there are any remaining tweets, group them together
            if num_tokens > 0:
                joined_tweets = " ".join(collected)
                samples.append(
                    self._prepare_sample(label, joined_tweets, num_tokens, user_data)
                )

        processed_df = pd.DataFrame.from_records(samples)

        # store as csv
        if {"gender", "ideology_binary", "ideology_multiclass"}.issubset(
            self._df.columns
        ):
            df_out_path = (
                Path(__file__).parent.parent
                / f"datasets/preprocessed_train_{self._lang}.csv"
            )
        else:
            df_out_path = (
                Path(__file__).parent.parent
                / f"datasets/preprocessed_test_{self._lang}.csv"
            )
        processed_df.to_csv(df_out_path, index=False)

        return processed_df
