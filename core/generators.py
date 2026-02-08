"""
Lead Generators for different categories
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid
from dataclasses import dataclass

from faker import Faker
from config.settings import COUNTRY_EMAIL_DOMAINS

@dataclass
class BaseGenerator:
    """Base generator class"""
    lead_type: str
    country: str
    fake: Faker
    
    def __init__(self, lead_type: str, country: str):
        self.lead_type = lead_type
        self.country = country
        
        # Set locale for Faker
        locale_map = {
            'DE': 'de_DE', 'CH': 'de_CH', 'NL': 'nl_NL',
            'US': 'en_US', 'UK': 'en_GB', 'FR': 'fr_FR',
            'IT': 'it_IT', 'ES': 'es_ES', 'JP': 'ja_JP'
        }
        locale = locale_map.get(country, 'en_US')
        self.fake = Faker(locale)
    
    def generate(self) -> Dict[str, Any]:
        """Generate lead data"""
        raise NotImplementedError
    
    def _generate_email(self, first_name: str, last_name: str) -> str:
        """Generate country-specific email"""
        domains = COUNTRY_EMAIL_DOMAINS.get(self.country, ["gmail.com"])
        domain = random.choice(domains)
        
        # Generate email with variations
        patterns = [
            f"{first_name.lower()}.{last_name.lower()}",
            f"{first_name.lower()}{last_name.lower()}",
            f"{first_name.lower()}_{last_name.lower()}",
            f"{first_name.lower()[0]}{last_name.lower()}"
        ]
        
        username = random.choice(patterns)
        return f"{username}@{domain}"
    
    def _generate_phone(self) -> str:
        """Generate country-specific phone number"""
        # Faker will generate locale-appropriate numbers
        return self.fake.phone_number()
    
    def _generate_name(self) -> tuple:
        """Generate culturally appropriate name"""
        return self.fake.first_name(), self.fake.last_name()

class HotLeadGenerator(BaseGenerator):
    """Generate hot leads"""
    
    def generate(self) -> Dict[str, Any]:
        first_name, last_name = self._generate_name()
        
        lead_data = {
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}",
            "email": self._generate_email(first_name, last_name),
            "phone": self._generate_phone(),
            "city": self.fake.city(),
            "state": self.fake.state() if hasattr(self.fake, 'state') else "",
            "country": self.country,
            "age": random.randint(25, 65),
            "occupation": self.fake.job(),
            "lead_type": self.lead_type,
            "lead_category": "hot",
            "status": "new",
            "priority": random.randint(1, 5),
            "source": "deepseek_ai",
            "generated_date": datetime.now().isoformat(),
            "signup_date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "last_activity_date": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
        }
        
        # Add type-specific fields
        if self.lead_type == "forex":
            lead_data.update(self._generate_forex_fields())
        elif self.lead_type == "crypto":
            lead_data.update(self._generate_crypto_fields())
        elif self.lead_type == "casino":
            lead_data.update(self._generate_casino_fields())
        
        return lead_data
    
    def _generate_forex_fields(self) -> Dict[str, Any]:
        """Generate Forex-specific fields"""
        return {
            "trading_experience": f"{random.randint(1, 20)} years",
            "account_size_usd": random.choice([1000, 5000, 10000, 50000, 100000]),
            "broker": random.choice(["MetaTrader", "cTrader", "NinjaTrader", "TradingView"]),
            "platform": random.choice(["MT4", "MT5", "cTrader", "Web Platform"]),
            "currency_pair": random.choice(["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]),
            "average_monthly_trades": random.randint(10, 500),
            "trading_strategy": random.choice(["Scalping", "Day Trading", "Swing Trading"]),
            "indicators_used": random.choice(["Moving Averages", "RSI", "MACD", "Bollinger Bands"]),
            "loss_amount_usd": random.randint(1000, 50000),
            "recovery_urgency": random.choice(["Low", "Medium", "High", "Critical"]),
            "funds_available_for_recovery": random.randint(1000, 50000)
        }
    
    def _generate_crypto_fields(self) -> Dict[str, Any]:
        """Generate Crypto-specific fields"""
        return {
            "victim_type": random.choice(["Individual", "Business", "Investor"]),
            "scam_platform": random.choice(["Fake Exchange", "Ponzi Scheme", "ICO", "Wallet Hack"]),
            "exchange_name": random.choice(["Binance", "Coinbase", "Kraken", "KuCoin"]),
            "cryptocurrency": random.choice(["Bitcoin", "Ethereum", "Ripple", "Cardano"]),
            "scam_type": random.choice(["Phishing", "Rug Pull", "Exit Scam", "Fake Investment"]),
            "total_loss_amount": random.randint(5000, 100000),
            "recovery_attempted": random.choice([True, False]),
            "evidence_available": random.choice([True, False]),
            "police_report_filed": random.choice([True, False]),
            "urgency_level": random.choice(["Low", "Medium", "High", "Critical"]),
            "fraud_category": random.choice(["Investment Scam", "Romance Scam", "Phishing Attack"])
        }
    
    def _generate_casino_fields(self) -> Dict[str, Any]:
        """Generate Casino-specific fields"""
        return {
            "monthly_income": random.choice([2000, 5000, 10000, 15000, 20000]),
            "gambling_frequency": random.choice(["Daily", "Weekly", "Monthly", "Occasionally"]),
            "favorite_games": random.choice(["Slots", "Blackjack", "Roulette", "Poker", "Baccarat"]),
            "casino_platform": random.choice(["Bet365", "888 Casino", "William Hill", "LeoVegas"]),
            "total_deposits": random.randint(1000, 50000),
            "total_withdrawals": random.randint(0, 20000),
            "net_loss": random.randint(1000, 30000),
            "biggest_single_loss": random.randint(500, 10000),
            "self_excluded": random.choice([True, False]),
            "chargeback_attempted": random.choice([True, False]),
            "addiction_level": random.choice(["Low", "Medium", "High", "Severe"]),
            "jurisdiction": random.choice(["Malta", "UK", "Gibraltar", "Curacao"])
        }

class RecoveryLeadGenerator(BaseGenerator):
    """Generate recovery leads"""
    
    def generate(self) -> Dict[str, Any]:
        first_name, last_name = self._generate_name()
        
        base_data = {
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}",
            "email": self._generate_email(first_name, last_name),
            "email_verified": random.choice([True, False]),
            "phone": self._generate_phone(),
            "phone_verified": random.choice([True, False]),
            "city": self.fake.city(),
            "state": self.fake.state() if hasattr(self.fake, 'state') else "",
            "country": self.country,
            "bio": self._generate_bio(),
            "signup_date": (datetime.now() - timedelta(days=random.randint(30, 730))).isoformat(),
            "last_activity_date": (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat(),
            "status": "recovery_seeking",
            "source": "deepseek_ai",
            "lead_type": self.lead_type,
            "lead_category": "recovery",
            "lead_source_link": self.fake.url(),
            "campaign_name": f"Recovery_{self.country}_{datetime.now().year}",
            "ad_id": f"AD{random.randint(10000, 99999)}",
            "utm_source": "google",
            "utm_medium": "cpc",
            "utm_campaign": f"recovery_{self.lead_type}",
            "utm_term": f"{self.lead_type}+recovery+services",
            "utm_content": "text_ad_v1",
            "interested_in": f"{self.lead_type} recovery services",
            "experience_years": random.randint(1, 10)
        }
        
        # Add type-specific recovery fields
        if self.lead_type == "forex":
            base_data.update(self._generate_forex_recovery_fields())
        elif self.lead_type == "crypto":
            base_data.update(self._generate_crypto_recovery_fields())
        elif self.lead_type == "casino":
            base_data.update(self._generate_casino_recovery_fields())
        
        return base_data
    
    def _generate_bio(self) -> str:
        """Generate recovery-specific bio"""
        bios = [
            f"Seeking recovery assistance after financial loss in {self.lead_type} trading.",
            f"Looking for professional help to recover funds from {self.lead_type} platform.",
            f"Victim of {self.lead_type} scam, need legal and recovery assistance.",
            f"Interested in recovery services for {self.lead_type} investment losses."
        ]
        return random.choice(bios)
    
    def _generate_forex_recovery_fields(self) -> Dict[str, Any]:
        """Generate Forex recovery fields"""
        return {
            "crypto_start_date": (datetime.now() - timedelta(days=random.randint(180, 1080))).isoformat(),
            "account_size": random.choice([5000, 10000, 25000, 50000, 100000]),
            "investment_type": random.choice(["Personal", "Business", "Retirement"]),
            "risk_tolerance": random.choice(["Low", "Medium", "High"]),
            "risk_level": random.choice(["Conservative", "Moderate", "Aggressive"]),
            "trading_style": random.choice(["Manual", "Automated", "Copy Trading"]),
            "trading_strategy": random.choice(["Trend Following", "Range Trading", "Breakout"]),
            "trading_frequency": random.choice(["Daily", "Weekly", "Monthly"]),
            "scam_type": random.choice(["Broker Fraud", "Signal Service", "Account Manager"]),
            "scam_date": (datetime.now() - timedelta(days=random.randint(7, 365))).isoformat(),
            "scam_details": f"Lost funds due to {random.choice(['manipulated spreads', 'withdrawal issues', 'fake broker'])}",
            "recovery_amount": random.randint(5000, 100000),
            "recovery_status": random.choice(["Not Started", "In Progress", "Legal Action"]),
            "broker": random.choice(["Unknown Broker", "Offshore Broker", "Unregulated Platform"]),
            "platform_name": random.choice(["MT4", "MT5", "WebTrader"]),
            "deposit_made": random.choice([True, False]),
            "deposit_amount": random.randint(1000, 50000) if random.choice([True, False]) else 0,
            "kyc_completed": random.choice([True, False]),
            "trade_type": random.choice(["CFD", "Spot", "Futures"]),
            "trade_pair": random.choice(["EUR/USD", "GBP/USD", "Gold/USD"]),
            "chargeback_attempted": random.choice([True, False]),
            "babypips_profile": f"user{random.randint(1000, 9999)}",
            "forexfactory_profile_id": f"ff_{random.randint(10000, 99999)}",
            "tradingview_profile_id": f"tv_{random.randint(10000, 99999)}"
        }
    
    def _generate_crypto_recovery_fields(self) -> Dict[str, Any]:
        """Generate Crypto recovery fields"""
        return {
            "crypto_start_date": (datetime.now() - timedelta(days=random.randint(180, 1080))).isoformat(),
            "account_size": random.randint(5000, 100000),
            "investment_type": random.choice(["Long-term", "Trading", "Staking"]),
            "risk_tolerance": random.choice(["Low", "Medium", "High"]),
            "scam_type": random.choice(["Exchange Hack", "Fake ICO", "Wallet Drainer", "Phishing"]),
            "scam_date": (datetime.now() - timedelta(days=random.randint(7, 365))).isoformat(),
            "scam_details": random.choice([
                "Fell for phishing email",
                "Fake exchange website",
                "Malicious smart contract",
                "Social engineering attack"
            ]),
            "recovery_amount": random.randint(5000, 250000),
            "recovery_status": random.choice(["Investigating", "Legal Process", "Blockchain Analysis"]),
            "exchange_name": random.choice(["FakeExchange", "ScamPlatform", "Unknown"]),
            "deposit_made": True,
            "deposit_amount": random.randint(1000, 50000),
            "kyc_completed": random.choice([True, False]),
            "trade_type": random.choice(["Spot", "Margin", "Futures"]),
            "cryptocurrency": random.choice(["BTC", "ETH", "USDT", "ADA", "SOL"]),
            "wallet_type": random.choice(["Hot Wallet", "Exchange Wallet", "Hardware Wallet"]),
            "wallet_address": f"0x{random.getrandbits(160):040x}",
            "chain_analysis": random.choice([True, False]),
            "chargeback_attempted": random.choice([True, False])
        }

class LeadGeneratorFactory:
    """Factory for creating lead generators"""
    
    @staticmethod
    def create_generator(generator_class: Any, lead_type: str, country: str) -> BaseGenerator:
        """Create a generator instance"""
        return generator_class(lead_type, country)

# Additional generator classes
class XDepositorGenerator(HotLeadGenerator):
    """Generate leads who have made deposits"""
    
    def generate(self) -> Dict[str, Any]:
        data = super().generate()
        data['lead_category'] = 'xdepositor'
        data['deposits_made'] = random.randint(1, 20)
        data['total_deposit_amount'] = random.randint(1000, 50000)
        data['last_deposit_date'] = (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        data['average_deposit'] = data['total_deposit_amount'] / data['deposits_made']
        return data

class NewTraderGenerator(HotLeadGenerator):
    """Generate new trader leads"""
    
    def generate(self) -> Dict[str, Any]:
        data = super().generate()
        data['lead_category'] = 'new_trader'
        data['experience_months'] = random.randint(1, 12)
        data['learning_resources'] = random.choice([
            "YouTube tutorials", "Online courses", "Mentorship", "Books"
        ])
        data['starting_capital'] = random.choice([500, 1000, 2000, 5000])
        data['goals'] = random.choice([
            "Learn basics", "Consistent profits", "Full-time trading", "Supplement income"
        ])
        return data

class OnlineScraperGenerator(BaseGenerator):
    """Generate leads from online scraping"""
    
    def generate(self) -> Dict[str, Any]:
        first_name, last_name = self._generate_name()
        
        platforms = {
            "forex": ["forexfactory.com", "babypips.com", "tradingview.com"],
            "crypto": ["bitcointalk.org", "reddit.com/r/cryptocurrency", "coinmarketcap.com"],
            "casino": ["askgamblers.com", "casinomeister.com", "reddit.com/r/gambling"]
        }
        
        source_platform = random.choice(platforms.get(self.lead_type, ["general_forum"]))
        
        return {
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}",
            "email": self._generate_email(first_name, last_name),
            "phone": self._generate_phone(),
            "city": self.fake.city(),
            "country": self.country,
            "lead_type": self.lead_type,
            "lead_category": "online_scraper",
            "source": "web_scraper",
            "source_platform": source_platform,
            "profile_url": f"https://{source_platform}/user/{first_name.lower()}",
            "scraped_date": datetime.now().isoformat(),
            "activity_level": random.choice(["Active", "Moderate", "Inactive"]),
            "post_count": random.randint(1, 1000),
            "join_date": (datetime.now() - timedelta(days=random.randint(30, 3650))).isoformat(),
            "last_seen": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            "interests": self._generate_interests(),
            "bio": self._generate_scraped_bio(),
            "website": self.fake.url() if random.random() > 0.7 else "",
            "social_profiles": self._generate_social_profiles()
        }
    
    def _generate_interests(self) -> List[str]:
        """Generate interests based on lead type"""
        interests_map = {
            "forex": ["Technical Analysis", "Fundamental Analysis", "Trading Strategies", "Risk Management"],
            "crypto": ["Blockchain", "DeFi", "NFTs", "Smart Contracts", "Mining"],
            "casino": ["Slot Machines", "Card Games", "Sports Betting", "Live Dealer"]
        }
        
        base_interests = interests_map.get(self.lead_type, ["Finance", "Investing"])
        return random.sample(base_interests, k=random.randint(2, len(base_interests)))
    
    def _generate_scraped_bio(self) -> str:
        """Generate bio for scraped lead"""
        bios = {
            "forex": [
                f"Forex trader interested in {random.choice(['price action', 'indicators', 'automation'])}",
                f"Sharing trading insights and market analysis",
                f"Learning and growing as a {self.lead_type} trader"
            ],
            "crypto": [
                f"Crypto enthusiast focused on {random.choice(['DeFi', 'Web3', 'Layer 2 solutions'])}",
                f"Blockchain developer and investor",
                f"Following the latest in cryptocurrency and blockchain technology"
            ],
            "casino": [
                f"Casino player sharing experiences and reviews",
                f"Interested in {random.choice(['bonus hunting', 'strategy games', 'live casinos'])}",
                f"Responsible gambling advocate"
            ]
        }
        
        return random.choice(bios.get(self.lead_type, [f"Interest in {self.lead_type}"]))
    
    def _generate_social_profiles(self) -> Dict[str, str]:
        """Generate social media profiles"""
        profiles = {}
        
        if random.random() > 0.3:
            profiles['twitter'] = f"@{self.fake.user_name()}"
        if random.random() > 0.5:
            profiles['facebook'] = f"facebook.com/{self.fake.user_name()}"
        if random.random() > 0.4:
            profiles['linkedin'] = f"linkedin.com/in/{self.fake.user_name().replace('.', '-')}"
        if random.random() > 0.6:
            profiles['telegram'] = f"@{self.fake.user_name()}"
        
        return profiles