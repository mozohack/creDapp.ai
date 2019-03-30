pragma solidity >=0.4.22 <0.6.0;
pragma experimental ABIEncoderV2;

// import "./SafeMath.sol";
import "https://raw.githubusercontent.com/oraclize/ethereum-api/master/oraclizeAPI_0.4.25.sol";

contract FraudFactory {
    
    address[] public allFrauds;
    uint256 public fraudCount;
    address public factoryOwner;
    
    function Factory() public {
        factoryOwner = msg.sender;
        fraudCount = 0;
    }
    
    function createContract() public {
        address newContract = new Fraud(factoryOwner, fraudCount++);
        allFrauds.push(newContract);
    }
    
    function getAllContracts() public view returns (address[]) {
        return allFrauds;
    }
    
    function getByID(uint256 queryID) public view returns (address) {
        return allFrauds[queryID];
    }
}
    

contract Fraud is usingOraclize {
    
    mapping (string => uint256) private intParams;
    mapping (string => string) private stringParams;
    mapping (string => bool) private boolParams;
    
    address fraudOwner;
    uint256 fraudID;
    
    string public fraudProbability;
    bool public isFraud; // take conclusion from fraudProbability on-chain
    
    string public paramURL;
    
    event LogConstructorInitiated(string nextStep);
    event LogOutputReceived(string output);
    event LogNewOraclizeQuery(string description);
    
    constructor(address fOwner, uint256 _fraudID) public {
        fraudOwner = fOwner;
        fraudID = _fraudID;
        LogConstructorInitiated("Constructor initiated. Oraclize is ready to make API requests.");
    }

    function () public { // solhint-disable-line
        // fallback function to disallow any other deposits to the contract
        revert();
    }

    function bytes32ToString(bytes32 x) constant returns (string) {
        bytes memory bytesString = new bytes(32);
        uint charCount = 0;
        for (uint j = 0; j < 32; j++) {
            byte char = byte(bytes32(uint(x) * 2 ** (8 * j)));
            if (char != 0) {
                bytesString[charCount] = char;
                charCount++;
            }
        }
        bytes memory bytesStringTrimmed = new bytes(charCount);
        for (j = 0; j < charCount; j++) {
            bytesStringTrimmed[j] = bytesString[j];
        }
        return string(bytesStringTrimmed);
    }
    
    function uintToBytes(uint v) constant returns (bytes32 ret) {
        if (v == 0) {
            ret = '0';
        }
        else {
            while (v > 0) {
                ret = bytes32(uint(ret) / (2 ** 8));
                ret |= bytes32(((v % 10) + 48) * 2 ** (8 * 31));
                v /= 10;
            }
        }
        return ret;
    }
    
    function getIntData(string attribute, uint256 val) public {
        intParams[attribute] = val;
    }
    
    function getStringData(string attribute, string val) public {
        stringParams[attribute] = val;
    }
    
    function getBoolData(string attribute, bool val) public {
        boolParams[attribute] = val;
    }
    
    function getParamURL() public view returns (string) {
        return paramURL;
    }
    
    function fraduProb() public view returns (string) {
        return fraudProbability;
    }
    
    function __callback(bytes32 myid, string result) {
        if (msg.sender != oraclize_cbAddress()) revert();
        fraudProbability = result;
        LogOutputReceived(result);
    }
    
    function getData(bytes32[] intAttr, uint256[] intVal, bytes32[] stringAttr, bytes32[] stringVal, bytes32[] boolAttr, bool[] boolVal) public returns (bool) {
        paramURL = "https://credapp--spotifyrounak.repl.co/predict?";
        bytes32 eq = "=";
        
        require(intAttr.length == intVal.length);
        require(stringAttr.length == stringVal.length);
        require(boolAttr.length == boolVal.length);
        
        for(uint i = 0; i < intAttr.length; i++) {
            getIntData(bytes32ToString(intAttr[i]), intVal[i]);
            paramURL = string(abi.encodePacked(paramURL, bytes32ToString(intAttr[i]), bytes32ToString(uintToBytes(intVal[i])), "&"));
        }
        
        for(uint j = 0; j < stringAttr.length; j++) {
            getStringData(bytes32ToString(stringAttr[j]), bytes32ToString(stringVal[j]));
            paramURL = string(abi.encodePacked(paramURL, bytes32ToString(stringAttr[j]), stringVal[j], "&"));
        }
        
        for(uint k = 0; k < boolAttr.length; k++) {
            getBoolData(bytes32ToString(boolAttr[k]), boolVal[k]);
            if(boolVal[k]) {
                paramURL = string(abi.encodePacked(paramURL, boolVal[k], "True", "&"));
            } else {
                paramURL = string(abi.encodePacked(paramURL, boolVal[k], "False", "&"));
            }
        }
        
        return true;
    }

    function computeFraud() payable {
        if (oraclize_getPrice("URL") > this.balance) {
            LogNewOraclizeQuery("Oraclize query was NOT sent, please add some ETH to cover for the query fee");
            oraclize_query("URL", string(abi.encodePacked(paramURL, "")));
        } else {
            LogNewOraclizeQuery("Oraclize query was sent, standing by for the answer..");
            
            // oraclize_query("URL", "json(https://api.kraken.com/0/public/Ticker?pair=ETHXBT).result.XETHXXBT.c.0");
            oraclize_query("URL", string(abi.encodePacked(paramURL, "")));
        }
    }
    
}