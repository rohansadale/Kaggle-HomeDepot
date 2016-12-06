# -*- coding: utf-8 -*-
import regex as re
class TextCleaner:

	numbers = [
			"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
			"eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
			"nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
		]
	digits = [
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90
		]

	text_cleaner_regex_dict = {
	'LowerUpperCaseSplitter_REGEX': [(r"(\w)[\.?!]([A-Z])", r"\1 \2"), (r"(?<=( ))([a-z]+)([A-Z]+)", r"\2 \3")], 
	'LetterLetterSplitter_REGEX' : [(r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2")],
	'HtmlCleaner_REGEX' : [(r"<.+?>", r""), (r"&nbsp;", r" "), (r"&amp;", r"&"), (r"&#39;", r"'"), (r"<br", r""),(r"gt/>", r""),
						(r"/>/Agt/>", r""), (r"</a<gt/", r"")],
	'DigitLetterSplitter_REGEX' : [(r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2"), (r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2")],
	'DigitCommaDigitMerger_REGEX' : [(r"(?<=\d+),(?=000)", r"")],
	'NumberDigitMapper_REGEX' : [(r"(?<=\W|^)%s(?=\W|$)"%n, str(d)) for n,d in zip(numbers, digits)],
	'UnitConverter_REGEX' : [(r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r"\1 in. "),
			(r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r"\1 lb. "),
			(r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r"\1 ft. "),
			(r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r"\1 sq.in. "),
			(r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 sq.ft. "),
			(r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r"\1 cu.in. "),
			(r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 cu.ft. "),
			(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. "),
			(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. "),
			(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. "),
			(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. "),
			(r"([0-9]+)( *)(minutes|minute)\.?", r"\1 min. "),
			(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. "),
			(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. "),
			(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. "),
			(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. "),
			(r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. "),
			(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr "),
			(r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. "),
			(r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr ")]
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