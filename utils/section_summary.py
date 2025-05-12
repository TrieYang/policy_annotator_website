import os
import time
import json
from dotenv import load_dotenv
import aiofiles

async def generate_section_summary(section_name, section_data, llm):
    """Generate a summary of compliance evaluation results for a specific model card section"""
    if TESTING_MODE:
        # Sample responses for testing mode
        sample_responses = {
                "System Name": """{
            "Overall": "#### Issues and Fixes:\\n- **Ambiguous system name**  \\n↳ *Clarify the system name to avoid confusion with other tools or versions.*\\n- **Missing version identifier**  \\n↳ *Include version number in the system name for better traceability.*\\n- **Inconsistent naming across documentation**  \\n↳ *Standardize system name usage across all technical and user documentation.*\\n- **No clear distinction from similar systems**  \\n↳ *Add unique identifiers to differentiate from related AI systems.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Missing unique identifier**  \\n↳ *Include a distinct and traceable system name or code to support regulatory filing.*\\n- **No conformity assessment reference**  \\n↳ *Link system name to EU conformity assessment documentation.*",
            "AIDA": "#### Issues and Fixes:\\n- **Inconsistent system labeling**  \\n↳ *Ensure the system name is uniform across all public and internal documentation.*\\n- **Missing risk classification indicator**  \\n↳ *Include risk level designation in system naming convention.*",
            "CCPA": "#### Issues and Fixes:\\n- **Name not tied to consumer-facing functionality**  \\n↳ *Clearly indicate which services use this AI system so consumers understand its presence.*\\n- **No privacy notice reference**  \\n↳ *Link system name to relevant privacy notices and data handling policies.*"
            }""",

                "Versioning Information": """{
            "Overall": "#### Issues and Fixes:\\n- **No version history provided**  \\n↳ *Add a versioning scheme and log of major changes to the system.*\\n- **Missing release dates**  \\n↳ *Document when each version was released and deployed.*\\n- **Incomplete change documentation**  \\n↳ *Detail significant changes between versions.*\\n- **No deprecation policy**  \\n↳ *Specify how long each version will be supported.*\\n- **Version compatibility unclear**  \\n↳ *Document compatibility between different versions.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Insufficient version traceability**  \\n↳ *Document each deployed version to support accountability during audits.*\\n- **No compliance version mapping**  \\n↳ *Link versions to specific regulatory compliance requirements.*",
            "AIDA": "#### Issues and Fixes:\\n- **No link between updates and risk**  \\n↳ *Explain how version updates are assessed for potential risk impacts.*\\n- **Missing impact assessment versions**  \\n↳ *Document which versions have undergone impact assessments.*",
            "CCPA": "#### Issues and Fixes:\\n- **Consumer-impacting updates unclear**  \\n↳ *Highlight changes that affect data use, privacy, or user-facing behavior.*\\n- **No data retention version policy**  \\n↳ *Specify how data retention periods vary across versions.*"
            }""",

                "Primary Developer/Org": """{
            "Overall": "#### Issues and Fixes:\\n- **Missing organization accountability**  \\n↳ *Specify the developing organization and responsible departments.*\\n- **Unclear development team structure**  \\n↳ *Document team composition and responsibilities.*\\n- **No contact hierarchy**  \\n↳ *Define escalation paths for different types of issues.*\\n- **Missing organizational chart**  \\n↳ *Provide clear reporting structure for the development team.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Provider identity vague**  \\n↳ *Clearly state the system provider and outline their responsibilities.*\\n- **No compliance officer designation**  \\n↳ *Identify the person responsible for EU AI Act compliance.*",
            "AIDA": "#### Issues and Fixes:\\n- **Lack of developer governance info**  \\n↳ *Include governance structures and responsible individuals for oversight.*\\n- **Missing risk management team**  \\n↳ *Specify who is responsible for risk assessment and mitigation.*",
            "CCPA": "#### Issues and Fixes:\\n- **Contact identity missing**  \\n↳ *Ensure users know who operates the system and how to reach them.*\\n- **No data protection officer**  \\n↳ *Designate a DPO for handling privacy-related inquiries.*"
            }""",

                "Contact Info": """{
            "Overall": "#### Issues and Fixes:\\n- **Missing support channel**  \\n↳ *Provide a clear contact for technical support and ethical concerns.*\\n- **Incomplete contact details**  \\n↳ *Include all necessary contact information for different types of inquiries.*\\n- **No response time commitments**  \\n↳ *Specify expected response times for different types of issues.*\\n- **Missing emergency contact**  \\n↳ *Provide 24/7 contact for critical system issues.*\\n- **No regional contact information**  \\n↳ *Include region-specific contact details for global support.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No incident reporting mechanism**  \\n↳ *Add a process for users to report potential harm or system failures.*\\n- **Missing regulatory contact**  \\n↳ *Provide contact information for EU regulatory authorities.*",
            "AIDA": "#### Issues and Fixes:\\n- **Accountability contact undefined**  \\n↳ *Specify a person or team responsible for compliance communications.*\\n- **No risk reporting contact**  \\n↳ *Designate a contact for reporting potential risks or harms.*",
            "CCPA": "#### Issues and Fixes:\\n- **No contact for data requests**  \\n↳ *Include a channel for users to request data access or deletion.*\\n- **Missing privacy inquiry contact**  \\n↳ *Provide dedicated contact for privacy-related questions.*"
            }""",
                "System Overview": """{
            "Overall": "#### Issues and Fixes:\\n- **High-level functionality unclear**  \\n↳ *Include a concise summary of what the AI system does and its boundaries.*\\n- **Missing system architecture**  \\n↳ *Provide a high-level overview of system components and their interactions.*\\n- **No deployment context**  \\n↳ *Specify where and how the system is deployed.*\\n- **Incomplete technical requirements**  \\n↳ *List hardware, software, and network requirements.*\\n- **Missing integration points**  \\n↳ *Document how the system integrates with other services.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Missing risk categorization**  \\n↳ *State whether the system qualifies as high-risk under Annex III of the EU AI Act.*\\n- **No conformity assessment scope**  \\n↳ *Define the scope of required conformity assessments.*",
            "AIDA": "#### Issues and Fixes:\\n- **No reference to intended impact**  \\n↳ *Describe the anticipated effects on individuals and society as required by AIDA.*\\n- **Missing risk level classification**  \\n↳ *Specify the system's risk level under AIDA framework.*",
            "CCPA": "#### Issues and Fixes:\\n- **Lacks mention of user data flow**  \\n↳ *Clarify how personal data moves through the system, if applicable.*\\n- **No data minimization statement**  \\n↳ *Explain how the system minimizes personal data collection.*"
            }""",

                "Primary intended uses": """{
            "Overall": "#### Issues and Fixes:\\n- **Intended use too vague**  \\n↳ *Clarify the real-world tasks or decisions the system is meant to support.*\\n- **Missing use case examples**  \\n↳ *Provide concrete examples of valid use cases.*\\n- **No performance expectations**  \\n↳ *Specify expected performance metrics for each use case.*\\n- **Incomplete success criteria**  \\n↳ *Define what constitutes successful system operation.*\\n- **Missing user workflow**  \\n↳ *Document typical user interactions and workflows.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No mapping to risk categories**  \\n↳ *Indicate whether any intended uses relate to high-risk applications under the regulation.*\\n- **Missing prohibited use cases**  \\n↳ *Explicitly list uses that are not permitted under EU regulations.*",
            "AIDA": "#### Issues and Fixes:\\n- **Societal impact not articulated**  \\n↳ *Describe how these use cases may influence people's rights or access to services.*\\n- **No equity considerations**  \\n↳ *Explain how the system ensures equitable access and outcomes.*",
            "CCPA": "#### Issues and Fixes:\\n- **Use cases lack privacy dimension**  \\n↳ *Explain how each use case relates to data collection or user profiling.*\\n- **Missing data purpose limitations**  \\n↳ *Specify how collected data will be used and stored.*"
            }""",

                "Primary intended users": """{
            "Overall": "#### Issues and Fixes:\\n- **User roles not defined**  \\n↳ *Specify who the system is designed for—experts, consumers, or institutions.*\\n- **Missing user qualifications**  \\n↳ *List required skills or training for system users.*\\n- **No user access levels**  \\n↳ *Define different user roles and their permissions.*\\n- **Incomplete user requirements**  \\n↳ *Specify technical and non-technical requirements for users.*\\n- **Missing user support needs**  \\n↳ *Document expected support requirements for different user types.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No human oversight mapping**  \\n↳ *Explain how user roles affect control over the AI system.*\\n- **Missing user responsibility definitions**  \\n↳ *Clarify user obligations under EU regulations.*",
            "AIDA": "#### Issues and Fixes:\\n- **Accessibility needs not considered**  \\n↳ *Account for inclusion of marginalized or underserved user groups.*\\n- **No user impact assessment**  \\n↳ *Document how different user groups may be affected.*",
            "CCPA": "#### Issues and Fixes:\\n- **End-user data rights undefined**  \\n↳ *Clarify how user roles affect data visibility or deletion options.*\\n- **Missing user consent requirements**  \\n↳ *Specify how user consent is obtained and managed.*"
            }""",

                "Out-of-scope use cases": """{
            "Overall": "#### Issues and Fixes:\\n- **No constraints listed**  \\n↳ *Document scenarios where system use is not advised or disallowed.*\\n- **Missing edge cases**  \\n↳ *Identify and document known edge cases and limitations.*\\n- **Incomplete risk scenarios**  \\n↳ *List potential misuse scenarios and their risks.*\\n- **No mitigation strategies**  \\n↳ *Provide guidance on handling out-of-scope situations.*\\n- **Missing system boundaries**  \\n↳ *Clearly define the limits of system capabilities.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No safeguards against misuse**  \\n↳ *Describe restrictions that prevent the AI system from being used in prohibited contexts.*\\n- **Missing prohibited use cases**  \\n↳ *List uses that violate EU regulations.*",
            "AIDA": "#### Issues and Fixes:\\n- **Potential for repurposing unacknowledged**  \\n↳ *List common misuses and provide warnings or disclaimers.*\\n- **No risk mitigation for misuse**  \\n↳ *Document measures to prevent harmful misuse.*",
            "CCPA": "#### Issues and Fixes:\\n- **No restriction on behavioral tracking**  \\n↳ *Clarify that system is not intended for unconsented behavioral analytics.*\\n- **Missing data use limitations**  \\n↳ *Specify prohibited data collection and processing activities.*"
            }""",

                "Terms and conditions": """{
            "Overall": "#### Issues and Fixes:\\n- **Missing licensing terms**  \\n↳ *Add legal conditions under which the system can be accessed and used.*\\n- **Incomplete liability clauses**  \\n↳ *Specify liability limitations and responsibilities.*\\n- **No warranty information**  \\n↳ *Document system warranties and guarantees.*\\n- **Missing termination conditions**  \\n↳ *Specify conditions for service termination.*\\n- **Incomplete dispute resolution**  \\n↳ *Provide process for handling disputes and complaints.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **User responsibilities undefined**  \\n↳ *Define who is responsible for compliance depending on deployment context.*\\n- **Missing regulatory compliance terms**  \\n↳ *Include specific EU AI Act compliance requirements.*",
            "AIDA": "#### Issues and Fixes:\\n- **No liability disclaimer**  \\n↳ *Include legal notices about AI-related harm and mitigation duties.*\\n- **Missing risk acceptance terms**  \\n↳ *Document user acceptance of system risks.*",
            "CCPA": "#### Issues and Fixes:\\n- **No consent-related language**  \\n↳ *State how consent is obtained or revoked under usage terms.*\\n- **Missing data rights terms**  \\n↳ *Specify terms related to data access and deletion rights.*"
            }""",

                "Current legal compliance status": """{
            "Overall": "#### Issues and Fixes:\\n- **No mention of applicable regulations**  \\n↳ *List which data, consumer, or AI regulations the system adheres to.*\\n- **Missing compliance documentation**  \\n↳ *Provide evidence of compliance with relevant regulations.*\\n- **Incomplete audit history**  \\n↳ *Document past compliance audits and their outcomes.*\\n- **No compliance roadmap**  \\n↳ *Outline plan for maintaining and improving compliance.*\\n- **Missing regulatory updates**  \\n↳ *Document how the system adapts to regulatory changes.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No reference to conformity assessment**  \\n↳ *Document status of required assessments under EU AI Act Title III.*\\n- **Missing risk classification**  \\n↳ *Specify the system's risk classification under EU AI Act.*",
            "AIDA": "#### Issues and Fixes:\\n- **Lack of risk-based compliance review**  \\n↳ *Note if the system underwent algorithmic impact assessment (AIA).*\\n- **Missing compliance timeline**  \\n↳ *Document compliance milestones and deadlines.*",
            "CCPA": "#### Issues and Fixes:\\n- **No disclosure of privacy policies**  \\n↳ *State whether the system has been reviewed for CCPA compliance.*\\n- **Missing data rights compliance**  \\n↳ *Document compliance with CCPA data rights requirements.*"
            }""",

                "Dataset Description": """{
            "Overall": "#### Issues and Fixes:\\n- **Data origin unclear**  \\n↳ *Provide source details and licensing terms of the datasets used.*\\n- **Missing data quality metrics**  \\n↳ *Include quality assessment results for training data.*\\n- **Incomplete data preprocessing**  \\n↳ *Document data cleaning and preparation steps.*\\n- **No data versioning**  \\n↳ *Track and document dataset versions.*\\n- **Missing data dependencies**  \\n↳ *List all data sources and their relationships.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No discussion of representativeness**  \\n↳ *Explain how datasets reflect the population where the system is used.*\\n- **Missing data governance**  \\n↳ *Document data management and oversight procedures.*",
            "AIDA": "#### Issues and Fixes:\\n- **Unclear data collection process**  \\n↳ *Indicate whether data was collected directly, indirectly, or scraped.*\\n- **No data impact assessment**  \\n↳ *Document assessment of data impact on different groups.*",
            "CCPA": "#### Issues and Fixes:\\n- **Lack of personal data flagging**  \\n↳ *Identify if any data used is considered personal under CCPA definitions.*\\n- **Missing data minimization**  \\n↳ *Explain how personal data collection is minimized.*"
            }""",

                "Collection Method": """{
            "Overall": "#### Issues and Fixes:\\n- **Unspecified collection procedure**  \\n↳ *Describe how and where data was collected, including devices or APIs used.*\\n- **Missing collection timeline**  \\n↳ *Document when and how long data was collected.*\\n- **Incomplete collection protocols**  \\n↳ *Detail standard operating procedures for data collection.*\\n- **No quality control measures**  \\n↳ *Describe quality assurance during data collection.*\\n- **Missing collection limitations**  \\n↳ *Document constraints and limitations in data collection.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Collection timeframe omitted**  \\n↳ *Specify when the data was obtained, as timing affects validity.*\\n- **No data governance documentation**  \\n↳ *Provide evidence of proper data management practices.*",
            "AIDA": "#### Issues and Fixes:\\n- **No consent method disclosed**  \\n↳ *Explain how user consent was obtained for data collection.*\\n- **Missing collection impact assessment**  \\n↳ *Document assessment of collection methods on different groups.*",
            "CCPA": "#### Issues and Fixes:\\n- **No opt-out capability for users**  \\n↳ *Document if users could reject participation or data tracking.*\\n- **Missing collection notice**  \\n↳ *Provide evidence of proper collection notices to users.*"
            }""",

                "Bias Mitigation Measures": """{
            "Overall": "#### Issues and Fixes:\\n- **Limited bias reduction explanation**  \\n↳ *Provide details on pre-processing, in-processing, or post-processing methods.*\\n- **Missing bias metrics**  \\n↳ *Include quantitative measures of bias reduction.*\\n- **Incomplete bias testing**  \\n↳ *Document testing procedures for bias detection.*\\n- **No ongoing monitoring**  \\n↳ *Describe continuous bias monitoring procedures.*\\n- **Missing bias incident response**  \\n↳ *Document procedures for handling bias-related incidents.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No fairness audits described**  \\n↳ *Include results from any bias testing required for high-risk systems.*\\n- **Missing bias risk assessment**  \\n↳ *Document assessment of bias risks under EU regulations.*",
            "AIDA": "#### Issues and Fixes:\\n- **No documentation of systemic risk**  \\n↳ *Explain whether the model was reviewed for disproportionate harm.*\\n- **Missing bias impact assessment**  \\n↳ *Document assessment of bias impact on different groups.*",
            "CCPA": "#### Issues and Fixes:\\n- **Protected group flags missing**  \\n↳ *Indicate if system treats sensitive data differently to mitigate bias.*\\n- **No bias-related data rights**  \\n↳ *Document how bias affects data access and correction rights.*"
            }""",

                "Usage Constraints": """{
            "Overall": "#### Issues and Fixes:\\n- **System limits not defined**  \\n↳ *Clarify technical and policy constraints on system usage.*\\n- **Missing performance boundaries**  \\n↳ *Document system performance limitations.*\\n- **Incomplete error handling**  \\n↳ *Specify how system handles errors and edge cases.*\\n- **No resource constraints**  \\n↳ *Document system resource requirements and limits.*\\n- **Missing scalability limits**  \\n↳ *Specify system scaling boundaries and constraints.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No usage boundary conditions**  \\n↳ *Define operating conditions to prevent deployment beyond scope.*\\n- **Missing risk-based constraints**  \\n↳ *Document constraints based on risk classification.*",
            "AIDA": "#### Issues and Fixes:\\n- **Lack of policy-driven usage checks**  \\n↳ *Specify measures that prevent harmful overuse or repurposing.*\\n- **No usage impact assessment**  \\n↳ *Document assessment of usage impact on different groups.*",
            "CCPA": "#### Issues and Fixes:\\n- **No data use limitations documented**  \\n↳ *Clearly list what data may not be used or retained under CCPA.*\\n- **Missing usage consent requirements**  \\n↳ *Document how usage consent is obtained and managed.*"
            }""",

                "Summary of Performance Assessment": """{
            "Overall": "#### Issues and Fixes:\\n- **No baseline or benchmark provided**  \\n↳ *Add quantitative metrics with comparisons to industry baselines.*\\n- **Missing performance metrics**  \\n↳ *Include comprehensive performance measurements.*\\n- **Incomplete evaluation methodology**  \\n↳ *Document how performance was evaluated.*\\n- **No performance goals**  \\n↳ *Specify target performance metrics.*\\n- **Missing performance monitoring**  \\n↳ *Describe ongoing performance monitoring procedures.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Performance for critical tasks unclear**  \\n↳ *Report accuracy, robustness, and reliability for key functions.*\\n- **No risk-based performance assessment**  \\n↳ *Document performance assessment based on risk level.*",
            "AIDA": "#### Issues and Fixes:\\n- **Impact of errors unaddressed**  \\n↳ *Discuss how model performance affects individuals or groups.*\\n- **Missing performance impact assessment**  \\n↳ *Document assessment of performance impact on different groups.*",
            "CCPA": "#### Issues and Fixes:\\n- **No indication of error in personal data usage**  \\n↳ *Include performance metrics relevant to privacy and personalization.*\\n- **Missing data accuracy metrics**  \\n↳ *Document accuracy of personal data processing.*"
            }""",

                "Disaggregated Performance": """{
            "Overall": "#### Issues and Fixes:\\n- **Subgroup performance not reported**  \\n↳ *Present how the system performs across different demographic or user groups.*\\n- **Missing intersectional analysis**  \\n↳ *Include performance across multiple demographic factors.*\\n- **Incomplete performance breakdown**  \\n↳ *Provide detailed performance metrics for each subgroup.*\\n- **No performance disparities**  \\n↳ *Document and explain performance differences between groups.*\\n- **Missing mitigation strategies**  \\n↳ *Describe plans to address performance disparities.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No fairness performance shown**  \\n↳ *Include subgroup error rates as required for high-risk classification.*\\n- **Missing risk-based performance**  \\n↳ *Document performance across risk categories.*",
            "AIDA": "#### Issues and Fixes:\\n- **Equity implications not explored**  \\n↳ *Report whether performance gaps might cause harm or exclusion.*\\n- **No equity impact assessment**  \\n↳ *Document assessment of performance impact on equity.*",
            "CCPA": "#### Issues and Fixes:\\n- **Demographic bias impact not measured**  \\n↳ *Evaluate if personalization varies by age, race, or other protected attributes.*\\n- **Missing data quality by group**  \\n↳ *Document data quality metrics across demographic groups.*"
            }""",

                "Testing Contexts": """{
            "Overall": "#### Issues and Fixes:\\n- **Unclear test environment**  \\n↳ *List environments and inputs used during testing, including edge cases.*\\n- **Missing test coverage**  \\n↳ *Document test coverage across different scenarios.*\\n- **Incomplete test methodology**  \\n↳ *Detail testing procedures and protocols.*\\n- **No test results documentation**  \\n↳ *Provide comprehensive test results and analysis.*\\n- **Missing test validation**  \\n↳ *Document how test results were validated.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Real-world conditions missing**  \\n↳ *Simulate and document tests under intended operational settings.*\\n- **No risk-based testing**  \\n↳ *Document testing based on risk classification.*",
            "AIDA": "#### Issues and Fixes:\\n- **Testing fails to simulate harms**  \\n↳ *Use representative conditions that reflect potential risk.*\\n- **Missing harm testing**  \\n↳ *Document testing for potential harms.*",
            "CCPA": "#### Issues and Fixes:\\n- **Data flow not validated in tests**  \\n↳ *Ensure tests cover scenarios involving personal data handling.*\\n- **No privacy testing**  \\n↳ *Document testing of privacy protections.*"
            }""",

                "Evaluations for Edge Cases or Adversarial Inputs": """{
            "Overall": "#### Issues and Fixes:\\n- **Limited adversarial testing**  \\n↳ *Expand test cases to include abnormal, unexpected, or hostile inputs.*\\n- **Missing edge case coverage**  \\n↳ *Document testing of system boundaries and limits.*\\n- **Incomplete robustness testing**  \\n↳ *Detail testing of system resilience.*\\n- **No security testing**  \\n↳ *Document testing of system security.*\\n- **Missing failure mode analysis**  \\n↳ *Document analysis of system failure modes.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Robustness testing incomplete**  \\n↳ *Evaluate resilience to edge cases as part of high-risk system obligations.*\\n- **No risk-based testing**  \\n↳ *Document testing based on risk level.*",
            "AIDA": "#### Issues and Fixes:\\n- **Potential for harm under-tested**  \\n↳ *Identify how unusual inputs may cause discriminatory or dangerous outcomes.*\\n- **Missing harm testing**  \\n↳ *Document testing for potential harms.*",
            "CCPA": "#### Issues and Fixes:\\n- **Adversarial misuse affecting personal data unaddressed**  \\n↳ *Include tests for manipulative attacks that affect privacy or output.*\\n- **No privacy attack testing**  \\n↳ *Document testing of privacy protections against attacks.*"
            }""",

                "Potential Risks and Harms": """{
            "Overall": "#### Issues and Fixes:\\n- **No risk assessment framework provided**  \\n↳ *List foreseeable risks, likelihoods, and severity, with mitigation strategies.*\\n- **Missing risk categories**  \\n↳ *Document different types of risks.*\\n- **Incomplete risk analysis**  \\n↳ *Provide detailed analysis of each risk.*\\n- **No risk monitoring**  \\n↳ *Describe ongoing risk monitoring procedures.*\\n- **Missing risk response plans**  \\n↳ *Document plans for responding to risks.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Annex III risk types not mapped**  \\n↳ *Align identified risks with EU AI Act high-risk categories.*\\n- **No risk-based assessment**  \\n↳ *Document assessment based on risk level.*",
            "AIDA": "#### Issues and Fixes:\\n- **Individual rights impact vague**  \\n↳ *Explain how the system may affect autonomy, dignity, or social inclusion.*\\n- **Missing rights impact assessment**  \\n↳ *Document assessment of impact on individual rights.*",
            "CCPA": "#### Issues and Fixes:\\n- **No disclosure of potential data misuse**  \\n↳ *Describe possible misuse of personal data and safeguards in place.*\\n- **Missing privacy risk assessment**  \\n↳ *Document assessment of privacy risks.*"
            }""",

                "Actions taken": """{
            "Overall": "#### Issues and Fixes:\\n- **Mitigations not linked to identified risks**  \\n↳ *Clearly show how specific actions address known risks or deficiencies.*\\n- **Missing action timeline**  \\n↳ *Document when actions were taken.*\\n- **Incomplete action documentation**  \\n↳ *Provide detailed documentation of actions taken.*\\n- **No action effectiveness**  \\n↳ *Document effectiveness of actions taken.*\\n- **Missing follow-up actions**  \\n↳ *Document planned follow-up actions.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **Corrective actions not documented**  \\n↳ *Include risk reduction efforts taken during design and testing phases.*\\n- **No risk-based actions**  \\n↳ *Document actions based on risk level.*",
            "AIDA": "#### Issues and Fixes:\\n- **Unclear alignment with accountability obligations**  \\n↳ *Highlight ongoing governance or improvement steps tied to compliance.*\\n- **Missing accountability actions**  \\n↳ *Document actions to ensure accountability.*",
            "CCPA": "#### Issues and Fixes:\\n- **No action noted on user complaints**  \\n↳ *Mention any steps taken in response to access, correction, or deletion requests.*\\n- **Missing privacy actions**  \\n↳ *Document actions taken to protect privacy.*"
            }""",

                "Misuse Scenarios": """{
            "Overall": "#### Issues and Fixes:\\n- **Misuse potential not documented**  \\n↳ *Describe how the system could be exploited or misunderstood.*\\n- **Missing misuse categories**  \\n↳ *Document different types of misuse.*\\n- **Incomplete misuse analysis**  \\n↳ *Provide detailed analysis of each misuse scenario.*\\n- **No misuse monitoring**  \\n↳ *Describe ongoing misuse monitoring procedures.*\\n- **Missing misuse response plans**  \\n↳ *Document plans for responding to misuse.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No prohibited use warnings**  \\n↳ *Include disclaimers for banned or high-risk uses, such as surveillance.*\\n- **Missing risk-based misuse**  \\n↳ *Document misuse scenarios based on risk level.*",
            "AIDA": "#### Issues and Fixes:\\n- **Misuse detection strategies missing**  \\n↳ *Explain how the system monitors or responds to unintended applications.*\\n- **No misuse impact assessment**  \\n↳ *Document assessment of misuse impact.*",
            "CCPA": "#### Issues and Fixes:\\n- **Data misuse not discussed**  \\n↳ *Identify how unauthorized access or repurposing of personal data is prevented.*\\n- **Missing privacy misuse**  \\n↳ *Document misuse scenarios affecting privacy.*"
            }""",

                "Human Oversight": """{
            "Overall": "#### Issues and Fixes:\\n- **Oversight mechanisms poorly defined**  \\n↳ *Specify when and how humans can intervene in system operation.*\\n- **Missing oversight roles**  \\n↳ *Document different oversight roles and responsibilities.*\\n- **Incomplete oversight procedures**  \\n↳ *Detail oversight procedures and protocols.*\\n- **No oversight training**  \\n↳ *Document training requirements for oversight personnel.*\\n- **Missing oversight monitoring**  \\n↳ *Describe ongoing oversight monitoring procedures.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No oversight protocol for high-risk decisions**  \\n↳ *Establish human review and override paths for sensitive outcomes.*\\n- **Missing risk-based oversight**  \\n↳ *Document oversight based on risk level.*",
            "AIDA": "#### Issues and Fixes:\\n- **Unclear if humans have meaningful control**  \\n↳ *Document how human judgment supplements or monitors automation.*\\n- **No oversight impact assessment**  \\n↳ *Document assessment of oversight impact.*",
            "CCPA": "#### Issues and Fixes:\\n- **Lack of manual appeal process for users**  \\n↳ *Ensure users can request human handling of automated decisions.*\\n- **Missing privacy oversight**  \\n↳ *Document oversight of privacy-related decisions.*"
            }""",

                "Update Frequency": """{
            "Overall": "#### Issues and Fixes:\\n- **Update cadence not disclosed**  \\n↳ *Describe how often and under what conditions the model or system is updated.*\\n- **Missing update process**  \\n↳ *Document update procedures and protocols.*\\n- **Incomplete update testing**  \\n↳ *Detail testing procedures for updates.*\\n- **No update validation**  \\n↳ *Document how updates are validated.*\\n- **Missing update monitoring**  \\n↳ *Describe ongoing update monitoring procedures.*",
            "EU AI Act": "#### Issues and Fixes:\\n- **No monitoring triggers for retraining**  \\n↳ *Indicate how performance drift or data changes lead to updates.*\\n- **Missing risk-based updates**  \\n↳ *Document updates based on risk level.*",
            "AIDA": "#### Issues and Fixes:\\n- **No link between updates and risk**  \\n↳ *Explain how update decisions incorporate harm prevention logic.*\\n- **No update impact assessment**  \\n↳ *Document assessment of update impact.*",
            "CCPA": "#### Issues and Fixes:\\n- **Versioning not tied to data retention**  \\n↳ *Clarify whether updates reset or affect data collection timelines.*\\n- **Missing privacy updates**  \\n↳ *Document updates affecting privacy.*"
            }"""
            }

        
        return sample_responses.get(section_name, f"""#### ⚠️ {section_name} – No Evaluation Data
Note: No evaluation data was provided for this section.""")
    
    # non-testing mode
    async with aiofiles.open("prompt_summarize_by_section.txt", "r") as f:
        prompt_template = await f.read()

    # Format the evaluation results for the prompt
    evaluation_results = []
    for policy_name, policy_data in section_data.items():
        for article, score in policy_data['scores'].items():
            if score < 5:  # Only include non-compliant items
                description = policy_data['descriptions'].get(article, "No description available")
                evaluation_results.append({
                    "policy": policy_name,
                    "article": article,
                    "score": score,
                    "description": description
                })

    # If no evaluation results, return a specific message for empty results
    if not evaluation_results:
        return f"""#### 🟢 {section_name} – Fully Compliant
        No compliance issues were identified for this section. All evaluated criteria meet the requirements."""

    # Format the evaluation results as a string
    evaluation_str = json.dumps(evaluation_results, indent=2)

    # Prepare the prompt
    prompt = prompt_template.replace("{{SECTION_NAME}}", section_name)
    prompt = prompt.replace("{{EVALUATION_RESULT}}", evaluation_str)

    # Get summary from Claude
    response = llm.invoke(prompt)
    return response.content.strip()