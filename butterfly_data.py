#this file is just for data so my mainpython file doesnt get too cluttered. 


class_names = [
    'AFRICAN GIANT SWALLOWTAIL', 'ADONIS', 'AN 88', 'AMERICAN SNOOT', 'APOLLO', 'ATALA',
    'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BANDED WHITE', 'BECKERS WHITE', 
    'BLACK HAIRSTREAK', 'BLUE SPOT HAIRSTREAK', 'BLUE MORPHO', 'BROWN SIPROETA', 
    'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHEQUERED SKIPPER', 'CLEOPATRA', 
    'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED PEACOCK', 'COMMON WOOD-NYMPH', 
    'COPPER TAIL', 'CRAMER\'S ORANGE TIP', 'CRIMSON PATCH', 'DAINTY SULPHUR', 
    'DANNAUS GILIPPUS', 'EASTERN COMMA', 'EASTERN DAPPLE WHITE', 'EASTERN PIED PIERROT', 
    'ELBOWED PIERROT', 'GAEA', 'GOLD BANDED', 'GRAY HAIRSTREAK', 'GREAT EGGFLY', 
    'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 
    'IPHICLUS SISTER', 'ISIS', 'JASON', 'LARGE MARBLE', 'MANGROVE SKIPPER', 
    'METALMARK', 'MILBERTS TORTOISESHELL', 'MINO WING', 'MONARCH', 'MOURNING CLOAK', 
    'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 
    'PIPEVINE SWALLOW', 'PURPLE HAIRSTREAK', 'PURPLE LEAFWING', 'QUEEN', 'QUESTION MARK', 
    'RED ADMIRAL', 'RED COSTUM', 'RED POSTMAN', 'SCARCE SWALLOW', 'SILVER SPOT HAIRSTREAK', 
    'SILVER-SPOT SKIPPER', 'SMALL COPPER', 'SLEEPY ORANGE', 'SOUTHERN DOGFACE', 
    'SPICEBUSH SWALLOW', 'STRAIGHT OAKEDGE', 'STRIPED QUEEN', 'TROPICAL LEAFWING', 
    'TWO BARRED FLINDER', 'ULYSSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 
    'ZEBRA LONG WING'
]


# define interesting facts for each butterfly
butterfly_facts = {
    'AFRICAN GIANT SWALLOWTAIL': "This butterfly is the largest in Africa, with a wingspan reaching up to 9.8 inches. Known for its toxicity, it likely gets its toxins from its caterpillar diet of certain rainforest plants, which helps it ward off predators.",
    'ADONIS': "The Adonis Blue butterfly is known for its bright blue males and its interesting relationship with ants. During its caterpillar stage, it secretes honeydew to attract ants, which protect it from predators in return. These butterflies also prefer warm, chalky grasslands in Europe.",
    'AN 88': "The '88' butterfly, or Diaethria anna, is named for the striking '88' pattern on the underside of its wings, making it visually unique among butterflies.",
    'AMERICAN SNOOT': "The American Snoot is named for its elongated mouthparts, giving it a 'snooty' look. It has bold black and white markings that help it blend into its surroundings.",
    'APOLLO': "Apollo butterflies are adapted to cold mountainous areas in Europe and Asia and have large, translucent wings that help with camouflage against rocky backgrounds.",
    'ATALA': "The Atala butterfly was once thought to be extinct due to habitat loss but has rebounded. Its caterpillars feed on toxic plants, which makes the butterfly itself distasteful to predators.",
    'BANDED ORANGE HELICONIAN': "This butterfly is known for its bright orange and black banded wings, which act as a warning to predators that it may be poisonous.",
    'BANDED PEACOCK': "The Banded Peacock butterfly sports a stunning iridescent green band on its wings and is found primarily in South Asia, where it is highly territorial.",
    'BANDED WHITE': "The Banded White is notable for its distinctive white bands, which help it camouflage among flowers and foliage in its South American habitats.",
    'BECKERS WHITE': "Native to North America’s arid regions, Becker's White butterfly has striking green-bordered wings and prefers desert plants from the mustard family as host plants.",
    'BLACK HAIRSTREAK': "This butterfly is quite rare in the UK and can be found only in certain blackthorn-rich habitats. It relies on ants to protect its larvae.",
    'BLUE SPOT HAIRSTREAK': "Known for the blue spots on its wings, this Mediterranean butterfly is adapted to hot, dry environments and is mostly active in summer.",
    'BLUE MORPHO': "The Blue Morpho's vibrant blue color is not due to pigment but to the microscopic structure of its scales, which reflect light to create an iridescent effect.",
    'BROWN SIPROETA': "Also called the Malachite butterfly, its brown and green wing pattern provides camouflage in tropical forests, where it feeds on rotting fruit.",
    'CABBAGE WHITE': "This common butterfly is considered a pest to gardeners as its larvae feed on cabbage and other plants in the mustard family.",
    'CAIRNS BIRDWING': "One of Australia’s largest butterflies, the Cairns Birdwing is known for its vivid green and black wings and relies on specific rainforest vines as host plants.",
    'CHEQUERED SKIPPER': "Found in Scotland and parts of Europe, this butterfly prefers grassy, open woodlands and has a quick, darting flight pattern.",
    'CLEOPATRA': "The Cleopatra butterfly resembles a large Brimstone and can be distinguished by the male’s deep orange wing coloration, which helps attract females.",
    'CLODIUS PARNASSIAN': "Males patrol habitat to find females; after mating they attach a pouch to female to prevent multiple matings. Females lay single eggs scattered on the host plant. ", 
    'CLOUDED SULPHUR': "They have exceptionally long tongues that allow them to reach the nectar of even the longest and narrowest flowers. ", 
    'COMMON BANDED PEACOCK': "Banded Peacocks are active year-round—although the lifespan of an individual adult is only two weeks!", 
    'COMMON WOOD-NYMPH': "The common wood nymph tastes with its feet and hears with its wings. ", 
    'COPPER TAIL': "They are rapid fliers and are usually distinguished by iridescent wings. The male's forelegs are reduced, but the female's are fully developed.", 
    'CRAMER\'S ORANGE TIP': "Orange-tip caterpillars are cannibals, eating their own eggshell when they emerge and moving on to eat other orange-tip eggs nearby. Caterpillars pupate in July and overwinter as a pupa, emerging as butterflies the following spring.", 
    'CRIMSON PATCH': "Crimson Patch butterflies have bright red patches on their hind wings that serve as a warning to predators about their unpalatable taste.",
    'DAINTY SULPHUR': "Dainty Sulphurs are one of the smallest butterflies in North America and can survive in arid environments, making them highly adaptable.",
    'DANNAUS GILIPPUS': "The Danaus gilippus, also known as the Queen butterfly, resembles the Monarch but has darker, reddish-brown wings and exhibits a fascinating mating behavior with pheromone-releasing 'hair pencils.'",
    'EASTERN COMMA': "Eastern Commas are named for the comma-shaped silver mark on the underside of their wings, which helps them blend into leaf litter when resting.",
    'EASTERN DAPPLE WHITE': "This butterfly has a unique dappled pattern on its wings, helping it blend into dappled light environments in its habitat.",
    'EASTERN PIED PIERROT': "Known for its black and white 'pied' pattern, this small butterfly is common in Asian regions and prefers gardens and open areas.",
    'ELBOWED PIERROT': "The Elbowed Pierrot has distinctive dark markings on its white wings, which it displays while basking in sunlight on open ground.",
    'GAEA': "The Gaea butterfly is named after the Greek goddess of Earth, and is typically found in high-altitude forests where it feeds on minerals in the soil.",
    'GOLD BANDED': "Gold Banded butterflies have a distinctive golden band on their wings, and they often use mud-puddling to extract nutrients from the ground.",
    'GRAY HAIRSTREAK': "Gray Hairstreaks have a unique tail-like extension on their hindwings and a false head pattern to mislead predators.",
    'GREAT EGGFLY': "The Great Eggfly butterfly has brilliant iridescent blue spots on its wings, which are used to attract mates and ward off rivals.",
    'GREAT JAY': "The Great Jay butterfly is known for its large size and vivid blue color, found primarily in Southeast Asia’s forests.",
    'GREEN CELLED CATTLEHEART': "This butterfly has a striking green coloration on its body and is found in South American rainforests, where it is toxic to predators.",
    'GREY HAIRSTREAK': "Grey Hairstreaks are widespread across North America and use their false head markings to deter predators.",
    'INDRA SWALLOW': "The Indra Swallowtail is found in mountainous areas and has uniquely patterned wings to blend into rocky environments.",
    'IPHICLUS SISTER': "This butterfly has a beautiful orange and black wing pattern and is known for its rapid, erratic flight.",
    'ISIS': "Named after the Egyptian goddess, the Isis butterfly has a vibrant pattern and is usually found in dense tropical forests.",
    'JASON': "The Jason butterfly, known for its striking colors, often feeds on tree sap rather than nectar.",
    'LARGE MARBLE': "The Large Marble butterfly has distinctive white and green mottling on its wings, helping it camouflage among plants.",
    'MANGROVE SKIPPER': "Found near mangroves, this butterfly’s larvae feed exclusively on mangrove plants.",
    'METALMARK': "Metalmarks get their name from the metallic-looking spots on their wings, which they use to distract predators.",
    'MILBERTS TORTOISESHELL': "This butterfly’s brilliant orange and black markings help it blend into autumn foliage.",
    'MINO WING': "The Mino Wing butterfly has an elongated wing shape that aids its quick, darting flight.",
    'MONARCH': "Monarch butterflies are famous for their long migration from North America to central Mexico, covering thousands of miles.",
    'MOURNING CLOAK': "Mourning cloaks are believed to be the longest-lived butterfly species in North America, with some living nearly up to a year as adults! Mourning cloaks seem to have only four legs.",
    'ORANGE OAKLEAF': "The Orange Oakleaf butterfly’s closed wings look remarkably like a dead leaf, providing excellent camouflage.",
    'ORANGE TIP': "The Orange Tip butterfly is named for the bright orange patches on the male’s wingtips, a characteristic absent in females.",
    'ORCHARD SWALLOW': "The Orchard Swallowtail is one of Australia’s largest butterflies and is a pest in citrus orchards.",
    'PAINTED LADY': "Painted Ladies are known for their global migration, traveling long distances across continents.",
    'PAPER KITE': "Paper Kite butterflies have delicate, translucent wings and are known for their slow, graceful flight.",
    'PIPEVINE SWALLOW': "The Pipevine Swallowtail’s caterpillars feed on toxic pipevine plants, making the butterfly poisonous to predators.",
     'PURPLE HAIRSTREAK': "Purple Hairstreaks are usually found high in oak trees, where they feed on honeydew produced by aphids.",
    'PURPLE LEAFWING': "The Purple Leafwing butterfly has a vibrant purple color on its upper wings, but it resembles a leaf when its wings are closed, aiding in camouflage.",
    'QUEEN': "The Queen butterfly resembles the Monarch and feeds on nectar, using toxins from milkweed to protect itself from predators.",
    'QUESTION MARK': "The Question Mark butterfly has a silver comma-shaped mark on its underside and a unique jagged wing shape.",
    'RED ADMIRAL': "Red Admirals are known for their territorial behavior, often chasing away other butterflies from their chosen flowers.",
    'RED COSTUM': "The Red Costum butterfly has bright red markings, making it one of the most eye-catching species in its habitat.",
    'RED POSTMAN': "The Red Postman butterfly is unique because it can recognize specific host plants by sight, essential for laying eggs.",
    'SCARCE SWALLOW': "The Scarce Swallowtail has long tail-like extensions on its hindwings and is often seen gliding gracefully.",
    'SILVER SPOT HAIRSTREAK': "Silver Spot Hairstreaks have a metallic silver spot on each hindwing, which reflects light to distract predators.",
    'SILVER-SPOT SKIPPER': "The Silver-Spot Skipper is known for its rapid, darting flight and the silver spots that help it blend in among plants.",
    'SMALL COPPER': "The Small Copper butterfly’s vibrant orange and black wings help it warm up quickly in the sun, aiding in flight.",
    'SLEEPY ORANGE': "The Sleepy Orange butterfly gets its name from the faint eye markings that look like closed, sleepy eyes.",
    'SOUTHERN DOGFACE': "The Southern Dogface butterfly has a forewing pattern resembling a dog's face, hence its unusual name.",
    'SPICEBUSH SWALLOW': "The Spicebush Swallowtail is known for its 'eye spots' on its wings, which deter predators by mimicking a larger animal.",
    'STRAIGHT OAKEDGE': "The Straight Oakedge butterfly is often found around oak trees, where its caterpillars feed and grow.",
    'STRIPED QUEEN': "The Striped Queen butterfly (a variant of the Queen butterfly) is closely related to the Monarch but can be distinguished by its rich orange-brown wings and lack of distinct black veins. It has unique “hair-pencils” near the abdomen that males use to release pheromones during mating, a feature not present in Monarchs ", 
    'TROPICAL LEAFWING': "This butterfly has a remarkable camouflage ability. When its wings are closed, it resembles a dried leaf, helping it evade predators. This natural mimicry allows it to blend seamlessly into forest habitats", 
    'TWO BARRED FLINDER': "This butterfly is known for the distinct two bars on its wings, which aid in both identification and mimicry. It uses these markings to deter predators by mimicking more toxic species", 
    'ULYSSES': "Also known as the Blue Mountain Swallowtail, the Ulysses butterfly is famous for its vivid blue iridescent wings. This butterfly is often associated with rainforests in Australia and serves as a symbol of the region’s rich biodiversity", 
    'VICEROY': "The Viceroy butterfly is a well-known mimic of the Monarch butterfly, sharing similar orange and black coloring. Unlike Monarchs, Viceroys have an extra black line across their hind wings, which helps distinguish them. This mimicry confuses predators into thinking it’s as toxic as the Monarch ", 
    'WOOD SATYR': "This butterfly prefers shaded, wooded habitats and has earthy brown wings with eye spots that deter predators. The Wood Satyr’s quiet flight and subtle coloring make it a master of camouflage in forested areas ", 
    'YELLOW SWALLOW TAIL': "Known for its large, bright yellow wings with black tiger-like stripes, the Yellow Swallowtail is a widespread butterfly in North America. Its size and coloration make it a striking sight in gardens and fields", 
    'ZEBRA LONG WING': "This butterfly is unique because it can digest pollen as well as nectar, giving it a longer lifespan than most other butterflies. Its black and white striped wings resemble a zebra, which helps with camouflage in dappled light"
}