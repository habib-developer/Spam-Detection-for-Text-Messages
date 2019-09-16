using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Spam_Detection.Data
{
    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public string isSpam { get; set; }
    }
}
