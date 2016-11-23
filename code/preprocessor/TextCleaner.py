# -*- coding: utf-8 -*-
import regex as re
class TextCleaner:

	text_cleaner_regex_dict = {
	'LowerUpperCaseSplitter_REGEX': [(r"(\w)[\.?!]([A-Z])", r"\1 \2"), (r"(?<=( ))([a-z]+)([A-Z]+)", r"\2 \3")], 
	'LetterLetterSplitter_REGEX' : [(r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2")],
	'HtmlCleaner_REGEX' : [(r"<.+?>", r""), (r"&nbsp;", r" "), (r"&amp;", r"&"), (r"&#39;", r"'"), (r"<br", r""),(r"gt/>", r""),
						(r"/>/Agt/>", r""), (r"</a<gt/", r"")]
	}

	def __init__(self, sentence):
		self.sentence = sentence

	def transform(self):
		for key, regex_pattern in TextCleaner.text_cleaner_regex_dict.iteritems():
			for pattern, replace in regex_pattern:
				try:
					self.sentence = re.sub(pattern, replace, self.sentence)
				except:
					pass
		return re.sub(r"\s+", " ", self.sentence).strip()