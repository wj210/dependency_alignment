"""Domain -> application -> goal -> five concrete subtasks.

This proposed catalogue contains 720 scenarios. Cases and document genres are
chosen later. IDs follow catalogue order; preserve that order once cases exist.
"""

CATALOGUE = {
    "Software and data": {
        "E-commerce systems": {
            "Calculate checkout prices and adjustments": [
                "Write a function that calculates order totals from item prices, discounts, shipping, and taxes.",
                "Build a transformation that converts promotion rules into a consistent checkout configuration table.",
                "Compare alternative rounding methods using supplied checkout examples and produce a differences table.",
                "Draft an interface specification for applying store credits across partial payments and refunds.",
                "Create a calculation worksheet showing how returned items change discounts and the remaining order balance.",
            ],
            "Coordinate order fulfillment across systems": [
                "Write a query joining order lines, stock reservations, and shipment events into a fulfillment view.",
                "Implement a function that partitions order items into shipments using warehouse and carrier constraints.",
                "Create a field mapping between storefront delivery statuses and warehouse tracking codes.",
                "Build a dashboard specification showing delayed shipments, split orders, and pending inventory transfers.",
                "Draft an integration plan for exchanging order changes between the storefront and fulfillment service.",
            ],
            "Support review of potentially fraudulent orders": [
                "Write a script joining order, account, and payment records into a fraud-review table.",
                "Prepare a case comparison matrix from supplied transaction histories and customer contact records.",
                "Draft handoff notes that organize an order's transaction timeline and unresolved review questions.",
                "Calculate fraud-review queue volumes and waiting times from supplied case timestamps.",
                "Create a form specification for recording review findings and subsequent order decisions.",
            ],
        },
        "Internal tools": {
            "Organize incoming staff service requests": [
                "Write a function that assigns request categories using supplied descriptions and category definitions.",
                "Build a query that combines request histories and queue ownership into an outstanding-work table.",
                "Draft a form specification for collecting equipment requests with department and delivery details.",
                "Calculate weekly queue capacity from request volumes, completion times, and staff availability.",
                "Create a workflow diagram showing how requests move between intake, assessment, and completion.",
            ],
            "Reduce repeated administrative data entry": [
                "Implement a script that converts spreadsheet expense entries into the accounting system's import format.",
                "Write a function that fills document fields from a supplied employee directory record.",
                "Compare two contact exports and produce a table of additions, changes, and duplicate entries.",
                "Draft an interface specification for synchronizing room bookings with the shared calendar.",
                "Create an import worksheet mapping legacy team codes to the current organizational directory.",
            ],
            "Maintain shared internal reference information": [
                "Build a search index configuration for locating policies by topic, department, and publication date.",
                "Write a query identifying directory entries whose team names differ across source systems.",
                "Draft a migration plan for moving reference pages into a new knowledge base.",
                "Create a comparison table showing differences between current and proposed document tagging schemes.",
                "Implement a transformation that merges glossary entries while retaining their source references.",
            ],
        },
        "Reporting pipelines": {
            "Combine data sources into consistent reporting tables": [
                "Write a transformation joining sales exports and customer records with different identifier formats.",
                "Create a field mapping that reconciles regional date formats and reporting currencies.",
                "Implement a function that groups duplicate event records using supplied matching criteria.",
                "Produce a source comparison table describing differences in coverage, granularity, and refresh frequency.",
                "Draft a schema for a reporting table combining historical snapshots with newly received records.",
            ],
            "Calculate and explain published business metrics": [
                "Write a query calculating monthly retention from customer activity and subscription records.",
                "Build a calculation worksheet reconciling dashboard revenue with refunds and deferred transactions.",
                "Compare metric definitions across departments and propose a shared definition table.",
                "Create chart specifications showing how conversion rates vary across channels and reporting periods.",
                "Draft a plain-language explanation of how missing observations affect the reported engagement metric.",
            ],
            "Manage scheduled refreshes and changing data feeds": [
                "Create a dependency map linking source extracts, transformation jobs, and published reports.",
                "Write a script that detects schema changes between successive source-file headers.",
                "Draft a refresh schedule using source availability times and report publication deadlines.",
                "Calculate reporting delays from job timestamps and produce a bottleneck comparison table.",
                "Prepare a migration checklist for replacing a data feed while preserving existing report fields.",
            ],
        },
        "Research software": {
            "Prepare study datasets for analysis": [
                "Write a transformation that combines participant measurements from differently structured study exports.",
                "Create a data dictionary describing supplied variables, units, coding conventions, and derived fields.",
                "Implement a function that groups repeated observations into participant-level analysis records.",
                "Produce a comparison table of missing-data patterns across collection sites and study periods.",
                "Draft a schema mapping linking questionnaire versions to a common analysis dataset.",
            ],
            "Implement and compare study analysis methods": [
                "Write an analysis function that calculates grouped estimates from supplied observations and weighting rules.",
                "Build a notebook comparing model specifications using the same supplied study dataset.",
                "Create a parameter configuration file for running alternative analysis settings across study cohorts.",
                "Draft an interface specification separating dataset preparation from statistical estimation routines.",
                "Generate a chart specification comparing estimates across analysis methods and observation windows.",
            ],
            "Reproduce and share computational research outputs": [
                "Create an environment file listing the package versions used by an existing analysis project.",
                "Write a script that regenerates figure inputs from archived analysis tables.",
                "Draft a README explaining how supplied scripts, configurations, and result files connect.",
                "Build a provenance table linking published figures to their data extracts and analysis parameters.",
                "Prepare a migration plan for moving an analysis workflow from notebooks into scheduled scripts.",
            ],
        },
    },
    "Defensive cybersecurity": {
        "Security monitoring": {
            "Organize monitoring events into usable datasets": [
                "Write a parser that converts supplied authentication logs into a common event schema.",
                "Build a query joining endpoint alerts with host inventory and event timestamps.",
                "Create a field mapping between two monitoring products' event categories and severity labels.",
                "Implement a transformation that groups duplicate alerts using supplied correlation criteria.",
                "Produce a coverage table showing which event types are available from each monitored service.",
            ],
            "Improve the handling of recurring alert queues": [
                "Calculate alert volumes and review times by category from supplied queue records.",
                "Create a comparison table of recurring alert patterns across hosts and observation periods.",
                "Draft a triage worksheet organizing alert evidence, affected assets, and follow-up questions.",
                "Write a query that groups related alerts into review batches using supplied asset and time rules.",
                "Prepare a dashboard specification showing queue age, category distribution, and repeated alert clusters.",
            ],
            "Maintain monitoring configuration and coverage": [
                "Compare supplied logging configurations and produce a table of missing event sources.",
                "Draft a configuration change proposal for collecting additional events from an internal service.",
                "Build an inventory mapping monitoring rules to the log fields they require.",
                "Create a local fixture set for checking how a parser handles different event formats.",
                "Prepare a rollout checklist for moving an existing monitoring configuration between managed environments.",
            ],
        },
        "Vulnerability management": {
            "Consolidate asset and vulnerability findings": [
                "Write a transformation joining scanner exports with software inventory and asset ownership records.",
                "Create a field mapping reconciling package names and version formats across inventory sources.",
                "Build a query grouping duplicate findings by asset, package, and advisory identifier.",
                "Produce a comparison table of finding coverage across scanner exports and collection dates.",
                "Draft a data dictionary for tracking findings, affected components, remediation status, and evidence sources.",
            ],
            "Plan software updates across managed systems": [
                "Create an update-planning matrix using supplied advisories, application dependencies, and maintenance windows.",
                "Calculate expected maintenance capacity from asset counts, update durations, and scheduling constraints.",
                "Draft a change request for updating a managed package using supplied release notes and deployment information.",
                "Build a dependency diagram showing applications affected by a proposed shared-library update.",
                "Compare update options and produce a table of version requirements and application compatibility notes.",
            ],
            "Track remediation progress over time": [
                "Write a query comparing successive scan results to identify resolved and newly reported findings.",
                "Create a dashboard specification showing remediation age, scheduled updates, and outstanding assets.",
                "Prepare a progress report linking remediation tickets to recent inventory and scan records.",
                "Implement a function that calculates finding age using discovery dates and closure records.",
                "Build a reconciliation table for discrepancies between ticket completion and installed package versions.",
            ],
        },
        "Access reviews": {
            "Build a consistent inventory of entitlements": [
                "Write a transformation joining account exports, group memberships, and application role definitions.",
                "Create a field mapping between identity-directory groups and application-specific role names.",
                "Build a query listing inactive accounts and their current entitlements from supplied activity records.",
                "Produce a comparison table of account identifiers across directory and application exports.",
                "Draft a data dictionary describing entitlement records, ownership fields, and membership sources.",
            ],
            "Prepare periodic entitlement review materials": [
                "Create a review worksheet grouping accounts and entitlements by application and business unit.",
                "Write a query highlighting entitlement changes since the previous review snapshot.",
                "Prepare a comparison matrix linking current role assignments to supplied role descriptions.",
                "Calculate review workloads from account counts, application coverage, and assigned review groups.",
                "Draft instructions explaining how to interpret the fields in an entitlement review export.",
            ],
            "Track entitlement decisions and account transitions": [
                "Build a reconciliation table comparing recorded review decisions with subsequent account configuration exports.",
                "Write a function matching employee role changes to related entitlement review records.",
                "Create a workflow diagram for documenting account transfers between departments.",
                "Prepare a progress report on completed reviews and outstanding entitlement-change tickets.",
                "Implement a transformation combining account departure dates with scheduled deactivation records.",
            ],
        },
        "Incident response": {
            "Assemble evidence into incident timelines": [
                "Write a script normalizing timestamps across supplied endpoint, identity, and service logs.",
                "Create a timeline table joining observed events with asset identifiers and evidence references.",
                "Produce a comparison matrix of event sequences reported by different monitoring sources.",
                "Draft an incident brief organizing observed activity, affected services, and unanswered investigation questions.",
                "Build a visualization specification showing event relationships across hosts and observation windows.",
            ],
            "Coordinate incident-related service changes": [
                "Create a dependency map showing services affected by proposed containment changes in supplied records.",
                "Draft a service change ticket using supplied incident notes and infrastructure ownership information.",
                "Build a tracking worksheet for planned changes, implementation timestamps, and service status reports.",
                "Calculate incident support coverage from shift schedules and outstanding coordination tasks.",
                "Prepare a handoff note connecting current incident observations with scheduled follow-up work.",
            ],
            "Review restoration work and incident follow-up": [
                "Create a restoration checklist from supplied service dependencies and recovery status records.",
                "Write a query comparing pre-incident and restored service inventory snapshots.",
                "Prepare a retrospective timeline linking service interruptions, response actions, and recovery milestones.",
                "Build a follow-up matrix connecting identified monitoring gaps with proposed configuration changes.",
                "Calculate service interruption durations and produce a comparison table across affected applications.",
            ],
        },
    },
    "Healthcare": {
        "Clinic workflows": {
            "Coordinate appointment capacity and patient flow": [
                "Write a function assigning appointment slots using supplied availability and visit-duration categories.",
                "Calculate expected appointment capacity from room schedules and staffing availability.",
                "Build a query grouping appointment delays by clinic session and visit category.",
                "Create a workflow diagram showing check-in, room allocation, and appointment completion stages.",
                "Prepare a comparison table of scheduling options for recurring clinic sessions.",
            ],
            "Organize referral processing and follow-up": [
                "Write a transformation joining referral records with appointment bookings and document receipt dates.",
                "Create a queue worksheet listing referral status, required documents, and pending scheduling steps.",
                "Calculate referral waiting times from receipt, review, and booking timestamps.",
                "Draft a referral status message using supplied administrative records and communication templates.",
                "Build a dashboard specification showing referral volumes and incomplete administrative steps.",
            ],
            "Structure visit documentation for clinic operations": [
                "Create a field mapping between an existing visit form and a replacement record template.",
                "Write a function converting supplied visit records into the clinic's structured export format.",
                "Draft a visit-summary template separating provided observations, recorded plans, and administrative follow-up.",
                "Produce a comparison table of documentation fields used across clinic departments.",
                "Prepare a migration checklist for transferring archived visit documents into a new record system.",
            ],
        },
        "Pharmacy operations": {
            "Manage stock availability and replenishment": [
                "Write a query joining inventory, dispensing totals, and incoming deliveries into a stock-status table.",
                "Calculate replenishment quantities using supplied demand estimates, delivery intervals, and stocking rules.",
                "Create a dashboard specification showing low-stock items and upcoming inventory review dates.",
                "Build a transformation grouping stock records by product, batch, and expiry month.",
                "Prepare a comparison matrix of supplier delivery schedules and product availability.",
            ],
            "Reconcile product catalogue and transaction records": [
                "Create a field mapping linking supplier product codes to the pharmacy's internal item catalogue.",
                "Write a function standardizing package-size descriptions in supplied product records.",
                "Build a reconciliation table comparing recorded stock movements with inventory counts.",
                "Draft a data dictionary describing product identifiers, package units, and transaction categories.",
                "Compare catalogue exports and produce a table of changed descriptions and duplicate product entries.",
            ],
            "Coordinate dispensing service administration": [
                "Create a queue worksheet organizing prescription receipt, processing, and collection status records.",
                "Calculate service waiting times by appointment category from supplied administrative timestamps.",
                "Draft a collection notification using supplied service status and an approved message template.",
                "Build a workflow diagram linking prescription intake, administrative checks, and collection scheduling.",
                "Prepare a staffing comparison table using service volumes and scheduled opening hours.",
            ],
        },
        "Patient education": {
            "Explain supplied health information in accessible language": [
                "Rewrite a supplied clinic leaflet in plain language while retaining its stated instructions.",
                "Create a glossary explaining technical terms found in supplied appointment preparation materials.",
                "Draft a question-and-answer sheet using only the supplied patient information booklet.",
                "Build a comparison table explaining differences between two provided descriptions of a clinic service.",
                "Prepare a visual outline showing the sequence described in an existing care-information leaflet.",
            ],
            "Adapt existing materials to different communication formats": [
                "Convert an approved information leaflet into a short narrated presentation script.",
                "Draft a text-message sequence conveying supplied appointment preparation instructions and scheduling details.",
                "Create a webpage outline organizing an existing booklet into navigable topic sections.",
                "Prepare a terminology checklist for translating supplied patient materials into another language.",
                "Write an audio-friendly version of a supplied handout for people using spoken information.",
            ],
            "Help people prepare for clinic conversations": [
                "Create a question worksheet based on supplied visit goals and patient information materials.",
                "Draft an appointment preparation checklist from supplied administrative and clinic instructions.",
                "Organize provided symptom diary entries into a chronological table for a clinic discussion.",
                "Build a note-taking template for recording questions, explanations, and agreed next steps during appointments.",
                "Prepare a comparison worksheet organizing questions about options already described in provided clinic materials.",
            ],
        },
        "Health research": {
            "Prepare observational health datasets": [
                "Write a transformation combining study visit records with supplied participant and measurement identifiers.",
                "Create a data dictionary describing coded outcomes, observation windows, and measurement units.",
                "Build a query calculating data completeness by collection site and study period.",
                "Implement a function grouping repeated measurements into the study's specified observation windows.",
                "Produce a comparison table showing differences between two versions of the study dataset.",
            ],
            "Support study planning and administration": [
                "Calculate projected recruitment totals using supplied site capacities and enrollment schedules.",
                "Create a workflow diagram connecting participant enrollment, scheduled visits, and data submission stages.",
                "Draft a study operations checklist from supplied timelines and administrative requirements.",
                "Build a visit-tracking worksheet showing expected dates, completed forms, and outstanding administrative items.",
                "Prepare a comparison matrix of data collection approaches described in supplied study proposals.",
            ],
            "Summarize quantitative study findings": [
                "Write an analysis function calculating grouped descriptive statistics from supplied study measurements.",
                "Create chart specifications comparing reported outcomes across observation periods and participant groups.",
                "Draft a results-table explanation using supplied estimates, intervals, and variable definitions.",
                "Build a calculation worksheet translating reported rates into counts for the supplied study population.",
                "Prepare a comparison table organizing findings from provided studies by measure and observation window.",
            ],
        },
    },
    "Biology": {
        "Bioinformatics": {
            "Harmonize biological data and annotation records": [
                "Write a transformation joining expression tables with supplied gene identifiers and annotation records.",
                "Create a field mapping between gene identifier systems using provided correspondence tables.",
                "Implement a parser converting supplied sequence metadata files into a common table format.",
                "Build a data dictionary describing assay measurements, sample labels, and normalization metadata.",
                "Produce a comparison table of dataset coverage across annotation versions and sample groups.",
            ],
            "Compare supplied biological measurements and annotations": [
                "Write a function calculating grouped expression summaries from a supplied measurement matrix.",
                "Create a comparison matrix of annotation categories across provided gene lists.",
                "Build a query joining sample attributes with analysis results for a cohort comparison.",
                "Prepare a chart specification showing expression patterns across supplied sample groups.",
                "Draft an explanation of differences between provided analysis outputs and their parameter settings.",
            ],
            "Organize reproducible bioinformatics analyses": [
                "Create a configuration file linking provided dataset paths to analysis parameter settings.",
                "Write a script combining result tables from completed analysis runs into a shared summary.",
                "Build a provenance table connecting figures with input datasets and software versions.",
                "Draft a README explaining the order of existing preprocessing and analysis scripts.",
                "Prepare a migration checklist for moving a completed analysis workflow to a different computing environment.",
            ],
        },
        "Ecological monitoring": {
            "Combine field observations into consistent records": [
                "Write a transformation joining species observations with site coordinates and survey dates.",
                "Create a field mapping reconciling species names and observation codes across supplied survey exports.",
                "Build a query identifying repeated observation records using supplied site and timestamp criteria.",
                "Draft a data dictionary describing habitat categories, survey effort, and observation units.",
                "Produce a comparison table of observation coverage across sites and sampling periods.",
            ],
            "Describe changes in monitored populations and habitats": [
                "Write a function calculating observation rates using supplied species counts and survey effort.",
                "Create chart specifications showing population observations across seasons and monitoring sites.",
                "Build a calculation worksheet comparing habitat-area estimates across supplied mapping surveys.",
                "Prepare a comparison matrix linking environmental measurements with recorded species observations.",
                "Draft an explanation of how uneven survey coverage affects the provided trend estimates.",
            ],
            "Plan and coordinate monitoring schedules": [
                "Create a survey schedule using supplied site access windows and field-team availability.",
                "Calculate fieldwork capacity from travel times, survey durations, and equipment availability.",
                "Build an equipment allocation worksheet for planned observation sessions across monitoring sites.",
                "Draft a workflow diagram linking field collection, record submission, and dataset consolidation.",
                "Prepare a comparison table of proposed survey schedules and their geographic coverage.",
            ],
        },
        "Laboratory experiments": {
            "Organize measurement plans for existing studies": [
                "Create a measurement schedule from supplied plant-growth study dates and instrument availability.",
                "Build an allocation worksheet linking existing sample groups to planned measurement sessions.",
                "Compare proposed observation schedules and produce a table of coverage and resource requirements.",
                "Draft a recording template for an existing microscopy study's measurements and image identifiers.",
                "Calculate the number of instrument sessions required for a supplied study measurement plan.",
            ],
            "Analyze supplied experimental measurements": [
                "Write a function calculating grouped summaries from supplied plant-growth and environmental measurements.",
                "Build a transformation combining instrument readings with sample identifiers and observation dates.",
                "Create chart specifications comparing recorded measurements across existing experimental groups.",
                "Prepare a calculation worksheet converting supplied instrument units into the study's reporting units.",
                "Draft an explanation comparing completed experiment results with the supplied measurement plan.",
            ],
            "Coordinate instrument use and measurement records": [
                "Create a booking schedule using instrument availability and supplied measurement-session requirements.",
                "Write a query matching completed measurement files to instrument booking records.",
                "Build a maintenance calendar from supplied instrument service dates and usage totals.",
                "Prepare a comparison table of file naming conventions used by different instruments.",
                "Draft a handoff note linking completed instrument sessions with outstanding data-processing work.",
            ],
        },
        "Sample management": {
            "Maintain a consistent sample catalogue": [
                "Write a transformation combining sample registration records with storage-location and study identifiers.",
                "Create a field mapping between legacy sample labels and the current catalogue format.",
                "Build a query identifying duplicate identifiers and unmatched registration entries in supplied records.",
                "Draft a data dictionary describing sample attributes, location codes, and record status fields.",
                "Produce a comparison table showing catalogue changes between successive inventory exports.",
            ],
            "Coordinate sample retrieval and record transfers": [
                "Create a retrieval worksheet organizing requested sample identifiers and recorded storage locations.",
                "Write a function grouping retrieval requests by study, storage unit, and requested date.",
                "Build a tracking table linking sample transfer requests with recorded receipt timestamps.",
                "Draft a workflow diagram connecting catalogue lookup, retrieval recording, and destination registration.",
                "Calculate retrieval workload from request counts and supplied storage-access schedules.",
            ],
            "Identify gaps in sample records and inventory": [
                "Build a reconciliation table comparing recorded sample locations with supplied inventory observations.",
                "Write a query listing samples without linked study records or completed registration fields.",
                "Create a dashboard specification showing catalogue completeness and unresolved inventory discrepancies.",
                "Prepare a review worksheet grouping missing record fields by study and storage unit.",
                "Draft an inventory review report summarizing supplied discrepancy records and outstanding catalogue updates.",
            ],
        },
    },
    "Chemistry": {
        "Analytical testing": {
            "Estimate sample concentrations": [
                "Fit a calibration curve from reference solutions and tabulated instrument responses.",
                "Calculate sample concentrations from peak areas, dilution factors, and calibration coefficients.",
                "Convert concentration results between reporting units using supplied sample properties.",
                "Estimate measurement uncertainty from replicate readings and calibration residuals.",
                "Create a calculation worksheet linking each concentration estimate to its sample identifier.",
            ],
            "Compare measurement methods": [
                "Compare two assay methods using paired concentration measurements and agreement plots.",
                "Summarize method detection limits from supplied blank and low-concentration measurements.",
                "Calculate recovery percentages for samples measured before and after reference additions.",
                "Prepare a comparison table of precision, turnaround time, and sample volume requirements.",
                "Explain differences between instrument methods using supplied chromatograms and method descriptions.",
            ],
            "Prepare batch result tables": [
                "Write a parser that combines instrument exports into a sample-level results table.",
                "Match chromatogram peak labels to analyte identifiers using a supplied reference table.",
                "Summarize replicate measurements by batch, including averages and observed variation.",
                "Create a batch summary showing missing measurements and available sample results.",
                "Format analytical results into a customer report using supplied reporting conventions.",
            ],
        },
        "Materials research": {
            "Compare material properties": [
                "Calculate elastic moduli from supplied stress-strain curves for candidate polymer films.",
                "Compare thermal conductivity measurements across materials with different specimen dimensions.",
                "Plot water absorption against exposure duration for supplied coating measurements.",
                "Prepare a material comparison table using measured properties and project requirements.",
                "Explain differences between candidate films using supplied composition and characterization summaries.",
            ],
            "Characterize aging behavior": [
                "Calculate property retention percentages from initial and aged specimen measurements.",
                "Fit a descriptive trend to coating thickness measurements collected over time.",
                "Compare surface images using supplied image descriptors and specimen exposure histories.",
                "Prepare a timeline connecting material changes to recorded environmental exposure conditions.",
                "Write an analysis plan for comparing aging results across specimen batches.",
            ],
            "Organize candidate screening": [
                "Rank candidate packaging materials using supplied property measurements and weighted selection criteria.",
                "Cluster material candidates by standardized mechanical and thermal property measurements.",
                "Create a screening matrix showing which properties have measurements for each candidate.",
                "Estimate characterization workload from candidate counts and available instrument schedules.",
                "Draft a comparison brief connecting candidate properties to packaging application requirements.",
            ],
        },
        "Process chemistry": {
            "Analyze process performance": [
                "Calculate mass balances from supplied inlet, outlet, and inventory measurements.",
                "Estimate separation recovery from feed compositions and collected fraction measurements.",
                "Plot batch yields against recorded process variables using historical production data.",
                "Create a dashboard summary of throughput, yield, and material consumption trends.",
                "Explain discrepancies between predicted and measured output using supplied process records.",
            ],
            "Compare operating scenarios": [
                "Compare supplied process simulations using energy consumption and product quality metrics.",
                "Calculate material requirements for alternative production volumes from established process ratios.",
                "Estimate utility costs for recorded operating scenarios using supplied tariff schedules.",
                "Prepare a tradeoff table comparing throughput, cost, and equipment utilization.",
                "Visualize sensitivity of simulation outputs to supplied parameter changes.",
            ],
            "Explain process variation": [
                "Align process sensor timestamps with batch records for historical variation analysis.",
                "Compare input material properties across batches with different recorded yields.",
                "Summarize associations between process deviations and recorded product quality measurements.",
                "Create a cause-and-observation diagram from supplied process investigation notes.",
                "Draft an investigation summary distinguishing recorded changes from proposed explanations.",
            ],
        },
        "Laboratory documentation": {
            "Organize experiment records": [
                "Convert supplied notebook entries into a structured experiment record template.",
                "Link raw measurement filenames to experiment identifiers in an index table.",
                "Prepare a timeline of instrument sessions from notebook and booking records.",
                "Summarize changes between two versions of an experiment's recorded method.",
                "Create a glossary translating instrument abbreviations used across supplied laboratory records.",
            ],
            "Prepare analytical reports": [
                "Draft a results section from supplied measurement tables and figure descriptions.",
                "Generate figure captions that identify samples, measured quantities, and units.",
                "Assemble a report appendix listing instrument settings from existing measurement records.",
                "Reformat analytical tables to match a supplied report template and unit convention.",
                "Prepare a short summary connecting each reported result to its experiment question.",
            ],
            "Coordinate sample records": [
                "Reconcile sample identifiers across intake forms, instrument exports, and archive records.",
                "Create a sample-location table from supplied storage and transfer records.",
                "Summarize remaining sample quantities from receipt records and logged withdrawals.",
                "Write a script that generates sample labels from a structured inventory table.",
                "Prepare a handoff list of samples awaiting measurements and their requested analyses.",
            ],
        },
    },
    "Mathematics and statistics": {
        "Forecasting": {
            "Prepare forecasting inputs": [
                "Aggregate irregular transaction records into weekly series with consistent time boundaries.",
                "Create lagged features from supplied demand and promotion history tables.",
                "Compare missing-value handling approaches using supplied time-series gaps and application requirements.",
                "Decompose a historical series into trend, seasonal, and residual components.",
                "Build a calendar feature table for supplied holidays and business operating days.",
            ],
            "Compare forecast estimates": [
                "Calculate forecast errors by horizon from supplied predictions and observed values.",
                "Compare seasonal and nonseasonal forecasts using rolling-window prediction results.",
                "Plot forecast intervals alongside observed demand for supplied evaluation periods.",
                "Summarize forecast performance separately for high-volume and low-volume product groups.",
                "Explain differences between forecast methods using supplied trend and seasonality diagnostics.",
            ],
            "Plan forecast updates": [
                "Calculate how updated demand observations change an existing moving-average forecast.",
                "Design a refresh schedule using supplied data arrival times and planning deadlines.",
                "Prepare a revision table comparing earlier forecasts with newly estimated values.",
                "Write a function that assembles forecast outputs into a planning workbook.",
                "Describe how a supplied structural change affects historical forecast assumptions.",
            ],
        },
        "Optimization": {
            "Formulate allocation problems": [
                "Translate staffing requirements and shift availability into a linear programming formulation.",
                "Define assignment variables and constraints for a supplied classroom scheduling problem.",
                "Build a transportation cost matrix from supplied origins, destinations, and shipment costs.",
                "Formulate a resource allocation objective using supplied project values and capacity limits.",
                "Create a constraint table linking each mathematical condition to a planning requirement.",
            ],
            "Compare feasible solutions": [
                "Calculate objective values for supplied allocation plans using the stated cost function.",
                "Compare scheduling alternatives using utilization, overtime, and unmet demand metrics.",
                "Construct a Pareto comparison of supplied cost and service-level solutions.",
                "Write a checker for capacity and assignment constraints in candidate schedules.",
                "Summarize solution changes when supplied resource limits increase or decrease.",
            ],
            "Explain optimization results": [
                "Translate a solver's allocation output into a department-level resource summary.",
                "Explain binding constraints using supplied solver results and problem definitions.",
                "Prepare a sensitivity table showing how objective values vary with demand assumptions.",
                "Visualize assignment results as a timetable using supplied solution variables.",
                "Draft a planning brief explaining tradeoffs among the supplied feasible alternatives.",
            ],
        },
        "Statistical studies": {
            "Prepare study analyses": [
                "Convert survey responses into analysis variables using a supplied coding guide.",
                "Calculate sample-size estimates from supplied effect sizes and study design assumptions.",
                "Create descriptive statistics tables for supplied participant groups and measurement variables.",
                "Specify an analysis model from the supplied research question and variable definitions.",
                "Prepare an analysis checklist linking study questions to available measurements.",
            ],
            "Estimate group differences": [
                "Calculate group mean differences and confidence intervals from supplied study measurements.",
                "Fit a regression model comparing groups while adjusting for supplied covariates.",
                "Estimate paired changes from participant measurements collected before and after an intervention.",
                "Compare weighted and unweighted estimates using supplied sampling weights.",
                "Create an effect-size plot summarizing estimates across supplied study subgroups.",
            ],
            "Explain study results": [
                "Translate supplied regression coefficients into examples using the study's measurement units.",
                "Draft a results paragraph from supplied estimates, intervals, and sample counts.",
                "Prepare a comparison table showing which findings vary across analysis specifications.",
                "Explain sampling variation through a simulation based on the supplied study design.",
                "Create figure captions describing group comparisons and the intervals shown.",
            ],
        },
        "Model validation": {
            "Assess predictive performance": [
                "Compute classification metrics from supplied predictions, labels, and decision thresholds.",
                "Plot calibration curves using supplied predicted probabilities and observed outcomes.",
                "Compare prediction errors across regions using supplied model outputs and observations.",
                "Calculate interval coverage from supplied prediction intervals and realized values.",
                "Summarize performance drift across successive evaluation periods using recorded model outputs.",
            ],
            "Examine model assumptions": [
                "Plot residual diagnostics for a regression using supplied fitted and observed values.",
                "Calculate feature correlations to examine overlap among supplied predictor variables.",
                "Compare residual distributions across supplied groups and measurement ranges.",
                "Analyze sensitivity of parameter estimates to specified transformations of input variables.",
                "Prepare an assumption review table from supplied diagnostics and model specifications.",
            ],
            "Plan model comparisons": [
                "Design time-ordered data splits for comparing models on a supplied forecasting dataset.",
                "Specify comparison metrics that match the supplied application's prediction requirements.",
                "Build a paired comparison table from supplied predictions on common observations.",
                "Estimate variability in metric differences using repeated supplied evaluation samples.",
                "Draft a comparison plan covering model versions, datasets, metrics, and reporting tables.",
            ],
        },
    },
    "Education": {
        "Lesson planning": {
            "Sequence learning activities": [
                "Arrange supplied lesson activities around prerequisite concepts and available classroom time.",
                "Draft a lesson timeline balancing explanation, guided practice, and independent work.",
                "Map each supplied classroom activity to the lesson's stated learning objectives.",
                "Design a transition activity connecting prior learning to a new concept.",
                "Prepare a lesson outline that builds from concrete examples to abstract notation.",
            ],
            "Prepare classroom materials": [
                "Create a worksheet using supplied example problems and target learning objectives.",
                "Draft slide explanations for a concept using the supplied lesson outline.",
                "Write discussion questions based on a supplied reading and classroom objective.",
                "Design a visual organizer showing relationships among the lesson's key ideas.",
                "Prepare a worked example using supplied problem details and grade-level conventions.",
            ],
            "Adapt lessons to class needs": [
                "Revise an activity for students with different levels of prerequisite knowledge.",
                "Adapt supplied lesson instructions for students learning the classroom language.",
                "Prepare extension activities for students who finish the supplied work early.",
                "Reschedule a lesson sequence after supplied changes to classroom time.",
                "Create alternative representations of a concept using the supplied learning materials.",
            ],
        },
        "Tutoring": {
            "Explain difficult concepts": [
                "Explain a fraction operation using a student's supplied attempted solution.",
                "Create an analogy for a scientific concept using the learner's stated interests.",
                "Compare two mathematical methods using a supplied problem and learner question.",
                "Break a supplied programming example into explanations of each logical step.",
                "Prepare a concept map connecting terms from a supplied reading passage.",
            ],
            "Plan guided practice": [
                "Generate practice questions progressing from supplied examples to a new problem type.",
                "Write a sequence of hints for a supplied problem at increasing detail levels.",
                "Prepare a short practice schedule using the learner's topics and available study time.",
                "Create a partially completed example for practicing a supplied mathematical procedure.",
                "Design a retrieval practice set using concepts from the learner's supplied notes.",
            ],
            "Respond to learning difficulties": [
                "Identify likely misconceptions from a supplied set of student problem attempts.",
                "Draft feedback connecting a student's written explanation to the target concept.",
                "Choose a follow-up question that distinguishes two explanations of a supplied error.",
                "Rewrite a difficult explanation using shorter steps and the learner's prior knowledge.",
                "Prepare a tutoring summary describing practiced concepts and questions still unresolved.",
            ],
        },
        "Assessment": {
            "Develop assessment questions": [
                "Write short-answer questions covering supplied objectives at specified difficulty levels.",
                "Create multiple-choice questions with distractors based on supplied student misconceptions.",
                "Design a practical assessment task using supplied course materials and time limits.",
                "Map draft assessment questions to a supplied topic coverage table.",
                "Revise a question whose wording conflicts with the supplied learning objective.",
            ],
            "Prepare grading materials": [
                "Draft a scoring rubric for a supplied project and its learning objectives.",
                "Write an answer guide explaining acceptable approaches to supplied assessment questions.",
                "Create annotated sample responses illustrating the levels in a supplied rubric.",
                "Prepare a marking spreadsheet from supplied rubric criteria and score ranges.",
                "Compare two draft rubrics for coverage of the supplied assignment requirements.",
            ],
            "Analyze assessment results": [
                "Calculate question-level success rates from a supplied table of student responses.",
                "Group assessment errors by concept using supplied responses and a topic map.",
                "Compare score distributions across classes using supplied assessment results.",
                "Prepare feedback summaries linking each student's results to the assessed concepts.",
                "Suggest review activities based on supplied class results and upcoming learning objectives.",
            ],
        },
        "Curriculum design": {
            "Map curriculum coverage": [
                "Map supplied course units to the program's stated learning outcomes.",
                "Identify repeated and uncovered concepts using supplied course outlines.",
                "Create a curriculum matrix showing when each skill is introduced and practiced.",
                "Compare two curricula using supplied topic lists and expected learner preparation.",
                "Summarize how supplied assessment activities cover the program's learning outcomes.",
            ],
            "Sequence curriculum units": [
                "Construct a prerequisite graph from supplied topics and course descriptions.",
                "Order supplied course units within the program's calendar and time limits.",
                "Estimate study workload by unit using supplied activities and expected completion times.",
                "Prepare a curriculum timeline connecting introductory concepts to later applied projects.",
                "Compare alternative unit sequences using supplied prerequisites and resource availability.",
            ],
            "Coordinate curriculum revisions": [
                "Summarize changes between supplied versions of a course outline.",
                "Prepare a revision table linking proposed changes to supplied learner feedback.",
                "Draft a transition plan for students moving between supplied curriculum versions.",
                "Create a resource update checklist from the supplied revised course objectives.",
                "Write a curriculum briefing describing revised topics, assessment changes, and teaching requirements.",
            ],
        },
    },
    "Engineering": {
        "Mechanical design": {
            "Compare design options": [
                "Compare candidate bracket designs using supplied dimensions, loads, and manufacturing costs.",
                "Prepare a weighted selection matrix for supplied enclosure design alternatives.",
                "Estimate component mass from supplied geometries and material density values.",
                "Compare fastening options using supplied assembly requirements and component specifications.",
                "Draft a design comparison brief explaining supplied packaging and assembly tradeoffs.",
            ],
            "Analyze component performance": [
                "Calculate beam deflection from supplied dimensions, material properties, and loading assumptions.",
                "Estimate thermal expansion for a component using supplied temperatures and material coefficients.",
                "Plot simulation outputs against measured displacement for supplied component trials.",
                "Calculate tolerance accumulation along a supplied assembly dimension chain.",
                "Prepare a sensitivity table showing how component dimensions affect the supplied performance metric.",
            ],
            "Prepare fabrication information": [
                "Create a dimension schedule from supplied component sketches and reference datums.",
                "Prepare a bill of materials from supplied assembly drawings and part lists.",
                "Draft assembly instructions using supplied component drawings and joining specifications.",
                "Compare drawing revisions and summarize changes to dimensions and material specifications.",
                "Generate an inspection table linking supplied dimensions to their specified tolerances.",
            ],
        },
        "Electronics": {
            "Analyze circuit behavior": [
                "Calculate resistor network voltages from supplied circuit topology and component values.",
                "Plot a filter's frequency response using supplied component values and circuit equations.",
                "Estimate a device's energy consumption from supplied operating currents and duty cycles.",
                "Compare simulated and measured waveforms for a supplied circuit configuration.",
                "Prepare a timing diagram from supplied signal transitions and interface specifications.",
            ],
            "Select circuit components": [
                "Compare voltage regulators using supplied load requirements and manufacturer specification tables.",
                "Calculate component value ranges for a supplied signal conditioning requirement.",
                "Prepare a component comparison table covering cost, package size, and electrical characteristics.",
                "Identify alternate components from supplied catalog entries and board footprint requirements.",
                "Estimate bill-of-materials costs for supplied circuit alternatives and production quantities.",
            ],
            "Prepare circuit documentation": [
                "Create a pin assignment table from supplied schematics and interface definitions.",
                "Draft a board bring-up checklist using supplied design notes and measurement points.",
                "Summarize schematic revisions affecting supplied connector assignments and signal names.",
                "Write an interface description using supplied voltage levels and timing requirements.",
                "Prepare a measurement worksheet linking supplied circuit nodes to expected signals.",
            ],
        },
        "Manufacturing": {
            "Plan production workflows": [
                "Create a production sequence from supplied operations, dependencies, and equipment availability.",
                "Calculate workstation workloads from supplied operation times and product mix.",
                "Prepare a material staging list for supplied production orders and assembly requirements.",
                "Build a shift schedule using supplied demand, staffing levels, and machine capacity.",
                "Draw a process flow diagram from supplied fabrication and assembly steps.",
            ],
            "Analyze product quality": [
                "Calculate dimensional variation from supplied inspection measurements across production batches.",
                "Create a control chart using supplied process measurements and sampling intervals.",
                "Summarize defect categories from supplied inspection notes and production quantities.",
                "Compare rejection rates before and after a recorded production process change.",
                "Prepare an investigation table connecting observed defects to supplied production records.",
            ],
            "Compare production alternatives": [
                "Estimate unit costs for supplied manufacturing routes and expected production volumes.",
                "Compare machine utilization under supplied production schedules and cycle times.",
                "Calculate changeover costs for alternative product sequencing plans from supplied records.",
                "Prepare a throughput comparison using supplied bottleneck times and batch sizes.",
                "Draft a manufacturing options brief covering supplied costs, capacity, and lead times.",
            ],
        },
        "Maintenance": {
            "Schedule maintenance work": [
                "Build a maintenance calendar from supplied service intervals and equipment operating hours.",
                "Estimate maintenance workloads using supplied task durations and available technician shifts.",
                "Prepare a spare-parts requirement list from supplied upcoming service tasks.",
                "Compare service scheduling options using supplied production windows and equipment availability.",
                "Create a maintenance work-order summary from supplied equipment records and scheduled tasks.",
            ],
            "Analyze equipment condition": [
                "Plot vibration trends from supplied measurements and recorded equipment operating conditions.",
                "Compare temperature readings across supplied operating periods and equipment units.",
                "Summarize recurring fault codes using supplied equipment logs and repair histories.",
                "Calculate operating time between recorded equipment failures from supplied service records.",
                "Prepare a condition summary connecting observed changes to supplied inspection findings.",
            ],
            "Prepare repair documentation": [
                "Turn supplied repair notes into a structured equipment service record.",
                "Create a parts replacement table from supplied work orders and component identifiers.",
                "Draft a repair handoff summarizing completed work and remaining recorded issues.",
                "Compare the recorded equipment configuration before and after a completed repair.",
                "Prepare a post-repair measurement checklist using supplied service instructions and observed faults.",
            ],
        },
    },
    "Finance": {
        "Accounting": {
            "Prepare the reporting-period close": [
                "Build a closing checklist from ledger accounts and the reporting calendar.",
                "Calculate accrual entries from service periods and unpaid invoice records.",
                "Create a depreciation schedule from asset registers and supplied accounting policies.",
                "Transform ledger balances into a trial balance with account groupings.",
                "Compile a period comparison table from current and prior financial statements.",
            ],
            "Organize and classify incoming expenses": [
                "Map invoice line descriptions to accounts using an existing chart of accounts.",
                "Write a parser that extracts expense dates, amounts, and currencies from exports.",
                "Calculate allocated expenses across departments using supplied allocation rules.",
                "Construct an invoice register linking source documents to expense categories.",
                "Compare expense coding across recurring invoices and identify changed classifications.",
            ],
            "Track outstanding customer receivables": [
                "Calculate invoice aging bands from due dates and recorded payment dates.",
                "Join customer receipts to invoice records using reference numbers and amounts.",
                "Prepare an outstanding-balance table separating invoices, credits, and partial payments.",
                "Build a receivables forecast from billing schedules and historical collection intervals.",
                "Draft account statements from supplied invoice and payment histories.",
            ],
        },
        "Budgeting": {
            "Allocate planned funding across activities": [
                "Build a budget worksheet from activity volumes and unit-cost assumptions.",
                "Calculate departmental allocations from a funding envelope and supplied weighting rules.",
                "Compare allocation proposals against category limits in the planning brief.",
                "Write a function that distributes shared costs across project budgets.",
                "Prepare a funding schedule linking planned expenditures to project milestones.",
            ],
            "Explain differences between planned and actual spending": [
                "Join budget lines to expenditure records using account and project codes.",
                "Calculate monthly variance tables with separate price and volume effects.",
                "Build a chart specification showing spending patterns across budget categories.",
                "Compare forecast revisions and identify assumptions driving changed spending estimates.",
                "Draft a variance explanation from transaction details and supplied operational notes.",
            ],
            "Compare future cost scenarios": [
                "Construct a scenario calculator varying demand, staffing levels, and recurring costs.",
                "Calculate break-even volumes for alternative service plans using supplied cost inputs.",
                "Prepare a sensitivity table for changes in selected budget assumptions.",
                "Compare phased and immediate spending plans over the same planning horizon.",
                "Create a rolling forecast that combines actual spending with remaining commitments.",
            ],
        },
        "Payments": {
            "Prepare payment batches from approved obligations": [
                "Transform approved invoice records into a payment-file layout with required fields.",
                "Calculate payable amounts after applying recorded credits and partial payments.",
                "Group payment instructions by currency, payment method, and processing date.",
                "Write a validator for payment-file field formats and supplied batch requirements.",
                "Build a payment calendar from invoice due dates and processing windows.",
            ],
            "Understand unsuccessful and delayed payments": [
                "Group payment failures by supplied provider response codes and processing stage.",
                "Construct an event timeline from payment attempts and provider status messages.",
                "Compare payment identifiers across request logs and provider transaction exports.",
                "Draft transaction-specific explanations from supplied payment statuses and service terms.",
                "Build a queue view separating unresolved attempts from completed payments.",
            ],
            "Track payment settlements and charges": [
                "Calculate settlement totals from transaction amounts, fees, refunds, and adjustments.",
                "Write a transformation joining provider settlements to internal payment records.",
                "Compare fee schedules against applied transaction charges in settlement exports.",
                "Create a settlement calendar from processing dates and provider timing rules.",
                "Prepare a table explaining settlement differences across currencies and payment methods.",
            ],
        },
        "Reconciliation": {
            "Match bank transactions to recorded cash activity": [
                "Write a matching function using references, amounts, and transaction dates.",
                "Calculate unmatched totals after linking bank statement lines to ledger entries.",
                "Transform bank exports into a common schema for multiple account formats.",
                "Prepare a reconciliation worksheet separating deposits, withdrawals, charges, and adjustments.",
                "Build date-window comparisons for receipts posted later than their bank transactions.",
            ],
            "Compare balances across internal financial records": [
                "Compare subsidiary ledger totals against control accounts for a reporting period.",
                "Join intercompany balances using entity codes and reciprocal account mappings.",
                "Calculate inventory-value differences between transaction records and accounting balances.",
                "Create a table aligning opening balances, movements, and closing balances.",
                "Write a transformation standardizing financial records from systems with different account codes.",
            ],
            "Explain reconciliation differences and their next processing steps": [
                "Construct a transaction timeline linking original entries, reversals, and later corrections.",
                "Group unmatched items by difference type using supplied reconciliation categories.",
                "Calculate the net effect of proposed adjustment entries on balance differences.",
                "Prepare item-specific notes connecting reconciliation differences to their source records.",
                "Build an outstanding-items table with recorded status and expected processing dates.",
            ],
        },
    },
    "Business and logistics": {
        "Inventory": {
            "Estimate stock replenishment needs": [
                "Calculate reorder quantities from current stock, demand estimates, and lead times.",
                "Build a demand profile from shipment records and seasonal planning inputs.",
                "Compare replenishment schedules under different order sizes and delivery intervals.",
                "Write a function projecting available stock after reservations and incoming deliveries.",
                "Prepare a purchase-needs table grouped by item, location, and expected shortage date.",
            ],
            "Maintain usable stock movement records": [
                "Transform receipts, transfers, and dispatches into a common inventory event schema.",
                "Calculate closing quantities from opening stock and recorded item movements.",
                "Join physical count results to recorded quantities using item and location identifiers.",
                "Construct a stock movement timeline for items with overlapping transfer records.",
                "Map product aliases and packaging units to the existing item catalogue.",
            ],
            "Improve inventory placement and handling plans": [
                "Compare storage locations using item dimensions, demand frequency, and capacity limits.",
                "Build a picking sequence from order lines and warehouse location data.",
                "Calculate space requirements for planned deliveries using packaging dimensions.",
                "Prepare a stock rotation list using receipt dates and supplied handling rules.",
                "Create a relocation plan linking proposed item moves to available storage capacity.",
            ],
        },
        "Procurement": {
            "Compare purchasing options against stated requirements": [
                "Build a supplier comparison table from quotations and requested product specifications.",
                "Calculate total purchase costs including delivery, recurring charges, and quantity discounts.",
                "Map quoted products to required specifications and identify unresolved differences.",
                "Compare delivery proposals against quantities and dates in the procurement plan.",
                "Prepare a scoring worksheet using supplied purchasing criteria and weights.",
            ],
            "Prepare purchase orders and supporting records": [
                "Generate purchase-order line items from approved selections and supplier catalogue entries.",
                "Write a transformation converting requisition exports into the purchasing system schema.",
                "Calculate order totals from quantities, unit prices, and supplied charge rules.",
                "Create an order checklist from required attachments and purchasing workflow stages.",
                "Draft order-specific delivery instructions from location details and receiving requirements.",
            ],
            "Monitor fulfillment of purchasing commitments": [
                "Join purchase orders to receipt records and calculate remaining ordered quantities.",
                "Construct a delivery-status table from supplier updates and scheduled receipt dates.",
                "Compare ordered items against shipment records using product and quantity mappings.",
                "Build a timeline linking order changes to updated delivery commitments.",
                "Calculate supplier delivery intervals from order dates and completed receipt records.",
            ],
        },
        "Customer support": {
            "Route incoming requests to the appropriate workflow": [
                "Classify support requests using a supplied issue taxonomy and routing rules.",
                "Extract order identifiers, requested actions, and relevant dates from incoming messages.",
                "Write a function grouping related tickets by customer and transaction references.",
                "Build an intake checklist tailored to information required for each request type.",
                "Prepare a queue view organized by issue category and recorded case status.",
            ],
            "Prepare useful responses for individual requests": [
                "Draft a response from the request details and supplied service policy.",
                "Create troubleshooting steps matched to the reported product configuration and symptoms.",
                "Calculate refund amounts from purchase records and supplied adjustment rules.",
                "Compare available replacement options against the customer's stated product requirements.",
                "Construct a case timeline linking prior contacts to the current request.",
            ],
            "Identify recurring service issues and workload patterns": [
                "Calculate contact volumes by issue category, product, and reporting period.",
                "Compare repeated requests with product releases and operational event dates.",
                "Transform ticket histories into resolution-duration data using recorded status transitions.",
                "Build a topic table from recurring customer descriptions and existing issue labels.",
                "Prepare a workload forecast from arrival patterns and average processing times.",
            ],
        },
        "Scheduling": {
            "Allocate appointment slots and shared resources": [
                "Build an appointment schedule from request durations and available resource windows.",
                "Write a function finding compatible time slots across supplied availability calendars.",
                "Calculate resource utilization from booked intervals and published operating hours.",
                "Compare appointment arrangements against travel times and location availability.",
                "Prepare rescheduling options after a resource becomes unavailable for specified intervals.",
            ],
            "Plan shifts against workload and availability": [
                "Calculate coverage requirements from workload forecasts and supplied staffing ratios.",
                "Build a shift roster from availability records and required skill combinations.",
                "Compare planned coverage with expected demand across time blocks.",
                "Write a validator for roster overlaps and supplied scheduling constraints.",
                "Prepare alternative shift arrangements after changes to availability and workload.",
            ],
            "Coordinate tasks with deadlines and dependencies": [
                "Construct a dependency graph from task descriptions and prerequisite relationships.",
                "Calculate earliest completion dates from durations and available working calendars.",
                "Build a milestone schedule linking deliverables to their required preceding tasks.",
                "Compare schedule options under alternative resource availability and task durations.",
                "Prepare a revised task sequence after a specified milestone changes date.",
            ],
        },
    },
    "Law and public administration": {
        "Legal research": {
            "Compare provisions in supplied legal texts": [
                "Build a comparison table of definitions across the supplied legal texts.",
                "Extract stated deadlines and triggering events into a provision reference table.",
                "Compare exceptions attached to similar requirements in supplied provisions.",
                "Map cross-references between supplied sections into a linked outline.",
                "Draft an explanation of how supplied provisions differ in stated scope.",
            ],
            "Organize supplied materials around a research question": [
                "Create an issue outline connecting the research question to supplied source passages.",
                "Build a chronology from events described in the provided case materials.",
                "Compare arguments in supplied decisions using a common issue framework.",
                "Prepare a source table separating stated facts, disputed facts, and addressed questions.",
                "Identify additional research questions arising from gaps in the supplied materials.",
            ],
            "Maintain usable research references": [
                "Standardize source citations using the supplied citation style and reference metadata.",
                "Write a parser extracting section identifiers from a supplied text collection.",
                "Build an annotated source index organized by the research outline.",
                "Compare repeated references across notes and consolidate entries for identical sources.",
                "Create a source-version table from supplied document dates and revision labels.",
            ],
        },
        "Document review": {
            "Extract document obligations and operative terms": [
                "Create a table of stated deliverables, deadlines, and triggering conditions.",
                "Map defined terms to their occurrences in the supplied document.",
                "Calculate milestone dates from supplied contract events and stated timing rules.",
                "Extract payment terms and prepare a schedule using supplied transaction dates.",
                "Build a clause index grouping provisions by their stated subject.",
            ],
            "Compare draft versions and related documents": [
                "Prepare a clause comparison showing additions, removals, and wording changes.",
                "Compare defined terms across a main document and its attachments.",
                "Write a script matching section identifiers between differently formatted document versions.",
                "Build a table of inconsistent dates and amounts across supplied documents.",
                "Create a change log linking revisions to supplied drafting comments.",
            ],
            "Assemble documents for a defined transaction workflow": [
                "Build a document checklist from transaction requirements and supplied template lists.",
                "Populate template fields from a supplied transaction information sheet.",
                "Prepare an attachment index matching referenced schedules to available files.",
                "Create a completion table from document statuses and outstanding drafting requests.",
                "Draft a cover explanation describing the contents of the assembled document set.",
            ],
        },
        "Service requests": {
            "Sort requests into defined public-service processes": [
                "Classify incoming requests using supplied service categories and routing rules.",
                "Extract application identifiers and requested services from submitted request descriptions.",
                "Build an intake checklist from published process steps supplied with the task.",
                "Map request locations to service areas using a supplied boundary table.",
                "Group related requests by location, subject, and submission period.",
            ],
            "Prepare request-specific responses and processing records": [
                "Draft a request response using supplied eligibility criteria and case facts.",
                "Calculate processing dates from recorded submissions and supplied service timeframes.",
                "Create a request timeline from submissions, correspondence, and recorded processing events.",
                "Prepare a missing-information checklist matched to the requested service type.",
                "Transform request records into the required case-management import format.",
            ],
            "Understand service volume and processing bottlenecks": [
                "Calculate request volumes across service categories and geographic areas.",
                "Build a process map from recorded stages and their stated dependencies.",
                "Compare completion times across submission channels using supplied processing records.",
                "Prepare a queue-aging table from open dates and current processing stages.",
                "Estimate processing capacity from arrival patterns and supplied stage durations.",
            ],
        },
        "Policy analysis": {
            "Compare options for a stated policy problem": [
                "Build an options table from supplied proposals and stated comparison criteria.",
                "Calculate estimated implementation costs using quantities and unit-cost assumptions.",
                "Map proposed measures to the stated problem and intended operational changes.",
                "Compare implementation timelines based on supplied prerequisites and available resources.",
                "Prepare a distribution table showing affected groups identified in supplied materials.",
            ],
            "Assess policy implementation using supplied records": [
                "Calculate implementation indicators from administrative records and supplied indicator definitions.",
                "Compare observed activity against milestones in the supplied implementation plan.",
                "Write a transformation linking program records across different reporting schemas.",
                "Build a geographic comparison table using supplied program and population data.",
                "Prepare a measurement plan linking policy questions to available data sources.",
            ],
            "Explain policy provisions to different intended audiences": [
                "Draft a plain-language explanation of requirements in the supplied policy text.",
                "Create a question-and-answer set from supplied policy provisions and recurring questions.",
                "Build a decision flowchart from stated conditions and process steps.",
                "Compare proposed communication drafts against the provided policy definitions.",
                "Prepare an example table illustrating supplied policy rules with fictional inputs.",
            ],
        },
    },
    "Social research and communication": {
        "Surveys": {
            "Design a questionnaire for stated research questions": [
                "Draft survey items mapped to the constructs in a supplied research brief.",
                "Build a questionnaire flow with branching based on prior responses.",
                "Compare response scales for the measurement goals in the survey plan.",
                "Prepare a pilot questionnaire checklist from supplied administration requirements.",
                "Calculate sample allocations across groups using supplied population counts and targets.",
            ],
            "Prepare survey responses for analysis": [
                "Write a transformation converting survey exports into a consistent analysis schema.",
                "Build a variable codebook from question labels and response categories.",
                "Calculate composite scores using supplied item mappings and scoring rules.",
                "Group free-text responses using an existing category framework.",
                "Prepare a response-completion table showing missing entries and skipped question paths.",
            ],
            "Answer research questions from survey results": [
                "Calculate weighted response distributions using supplied survey weights and group definitions.",
                "Compare response patterns across groups named in the analysis plan.",
                "Build a chart specification linking survey measures to research questions.",
                "Prepare an analysis table comparing repeated survey waves using common items.",
                "Draft a findings explanation from supplied estimates and the survey methodology.",
            ],
        },
        "Interview analysis": {
            "Organize transcript excerpts using an analysis framework": [
                "Build a coding table linking transcript excerpts to supplied category definitions.",
                "Write a parser splitting transcripts into speaker turns and timestamped segments.",
                "Prepare a codebook with examples drawn from supplied transcript passages.",
                "Compare overlapping codes and propose distinctions within the supplied framework.",
                "Create an excerpt index organized by topic and interview identifier.",
            ],
            "Compare experiences across interviews": [
                "Build a theme-by-interview matrix using coded excerpts and interview metadata.",
                "Compare accounts of a shared event across supplied interview transcripts.",
                "Construct a process timeline from events described across several interviews.",
                "Prepare a table contrasting themes across the stated participant groups.",
                "Calculate coding agreement from supplied annotations on the same transcript segments.",
            ],
            "Develop findings supported by the transcript collection": [
                "Draft theme descriptions linked to relevant excerpts from supplied interviews.",
                "Build an evidence table connecting proposed findings to coded passages.",
                "Compare a proposed explanation with excerpts describing differing experiences.",
                "Prepare a thematic outline answering the stated research question.",
                "Select illustrative excerpts for a findings section using supplied selection criteria.",
            ],
        },
        "Publication preparation": {
            "Structure a manuscript around the supplied work": [
                "Build a manuscript outline from the research question and supplied results.",
                "Draft a methods description from the study protocol and recorded procedures.",
                "Prepare an abstract from supplied manuscript sections and venue length limits.",
                "Compare section coverage against the supplied reporting checklist.",
                "Create a revision plan linking editorial comments to manuscript sections.",
            ],
            "Prepare tables, figures, and references for publication": [
                "Transform analysis outputs into tables following supplied publication formatting requirements.",
                "Write plotting code for supplied estimates and the requested figure layout.",
                "Draft figure captions explaining displayed measures and provided study context.",
                "Standardize references using supplied bibliographic records and the requested citation style.",
                "Build a cross-reference table matching manuscript mentions to figures and tables.",
            ],
            "Assemble and revise a submission package": [
                "Build a submission checklist from the supplied venue instructions and manuscript files.",
                "Draft a cover letter from the study description and submission requirements.",
                "Prepare a response table linking review comments to supplied manuscript revisions.",
                "Compare manuscript versions and produce a section-based revision log.",
                "Create a supplementary-material index matching referenced files to manuscript sections.",
            ],
        },
        "Public communication": {
            "Plan communication around a stated message and audience": [
                "Build a communication plan from audience descriptions and the stated campaign goals.",
                "Compare communication channels using supplied reach, timing, and resource information.",
                "Create a message map connecting key points to supporting supplied material.",
                "Prepare a publication calendar linked to planned announcements and available content.",
                "Draft an audience question list from supplied background and previous feedback.",
            ],
            "Adapt supplied content into usable communication formats": [
                "Draft a short announcement from supplied source material and format requirements.",
                "Create an explanatory script using the supplied facts and audience description.",
                "Prepare infographic text linking supplied quantities to their described meanings.",
                "Rewrite technical material using a supplied vocabulary and reading-level target.",
                "Build a question-and-answer page from supplied information and recurring audience questions.",
            ],
            "Interpret audience responses and communication results": [
                "Group audience questions by topic using a supplied communication framework.",
                "Calculate engagement measures from channel exports and supplied metric definitions.",
                "Compare audience responses across content versions and publication periods.",
                "Build a feedback table linking recurring questions to existing information resources.",
                "Prepare a content revision plan from supplied feedback and communication goals.",
            ],
        },
    },

}


def iter_scenarios():
    """Yield each scenario's ID and four specific context variables."""
    for domain_index, (domain, applications) in enumerate(CATALOGUE.items(), 1):
        for app_index, (application, goals) in enumerate(applications.items(), 1):
            for goal_index, (goal, subtasks) in enumerate(goals.items(), 1):
                for task_index, subtask in enumerate(subtasks, 1):
                    yield {
                        "scenario_id": (
                            f"d{domain_index:02d}_a{app_index:02d}_"
                            f"g{goal_index:02d}_s{task_index:02d}"
                        ),
                        "domain": domain,
                        "application": application,
                        "goal": goal,
                        "subtask": subtask,
                    }


if __name__ == "__main__":
    applications = sum(len(apps) for apps in CATALOGUE.values())
    goals = sum(len(goals) for apps in CATALOGUE.values() for goals in apps.values())
    scenarios = sum(1 for _ in iter_scenarios())
    print(f"{len(CATALOGUE)} domains, {applications} applications, "
          f"{goals} goals, {scenarios} scenarios")
