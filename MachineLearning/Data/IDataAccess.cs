using Microsoft.ML;
using Microsoft.ML.Runtime;

namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per l'accesso ai dati
   /// </summary>
   public interface IDataAccess : IChannelProvider, IDataView { }
}
