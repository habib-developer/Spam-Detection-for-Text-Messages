using Microsoft.AspNetCore.Mvc;
using Spam_Detection.Data;
using Spam_Detection.ML_Model;

namespace Spam_Detection.Controllers
{
    public class SpamDetectionController : Controller
    {
        public IActionResult Predict()
        {
            return View();
        }
        [HttpPost]
        public IActionResult Predict(SpamInput input)
        {
            var model = new SpamDetectionMLModel();
            model.Build();
            model.Train();
            ViewBag.Prediction = model.Predict(input);
            return View();
        }
    }
}
