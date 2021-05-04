use std::{fs::File, io::{BufRead, BufReader}, str::FromStr};
use strum_macros::EnumString;


#[derive(Debug, EnumString)]
pub enum LinkType {
    ID,
    NOT,
    OR,
    AND,
    XOR,
    ANY,
    ALL,
    EVEN,
    ODD,
    NOISY,
}

#[derive(Debug)]
pub struct LinkInfo {
    pub element: String,
    pub link_type: LinkType,
    pub condition: Vec<String>,
}

fn parse_sif_line(line: &str) -> LinkInfo {
    let mut iter = line.split(" ");

    let element = iter.next().expect("No element is defined");

    let link_type_str = iter.next().expect("No link is defined");

    let condition: Vec<String> = iter.map(|x| x.to_string()).collect();

    if condition.len() == 0 {
        panic!("No condition is defined")
    }

    LinkInfo {
        element: element.to_string(),
        condition: condition,
        link_type: LinkType::from_str(link_type_str).unwrap(),
    }
}

pub fn read_sif(path: &str) -> Vec<LinkInfo> {
    let file = File::open(path).unwrap();

    let mut infos = Vec::<LinkInfo>::new();

    for line in BufReader::new(file).lines() {
        let unwrapped = line.unwrap();
        let parsed = parse_sif_line(&unwrapped);
        infos.push(parsed);
    }

    infos
}
